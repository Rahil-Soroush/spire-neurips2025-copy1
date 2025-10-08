
import torch
from scipy.linalg import subspace_angles
import pandas as pd
import numpy as np
import os
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score, pairwise_distances, accuracy_score, precision_score, recall_score, f1_score
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier

from data.data_loader import build_dataset_with_lag


def as_3d(a):
    a = a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)
    if a.ndim == 2:  # (N, T) -> (N, T, 1)
        a = a[..., None]
    return a

def extract_latents_from_test_set_align(model, gpi_test, stn_test, device, batch_size=32):
    model.eval()
    out = {'shared_gpi': [], 'shared_stn': [], 'private_gpi': [], 'private_stn': []}
    out.update({'shared_gpi_aligned': [], 'shared_stn_aligned': []})

    # simple batching for speed
    N = gpi_test.shape[0]
    for i in range(0, N, batch_size):
        xb = gpi_test[i:i+batch_size].to(device)
        yb = stn_test[i:i+batch_size].to(device)
        with torch.no_grad():
            _, _, zgx, zgy, zpx, zpy = model(xb, yb, private_gate=1.0)
            out['shared_gpi'].append(zgx.cpu())
            out['shared_stn'].append(zgy.cpu())
            out['private_gpi'].append(zpx.cpu())
            out['private_stn'].append(zpy.cpu())
            # apply the same aligners used in training (conv + mapper unless identity flag is on)
            out['shared_stn_aligned'].append(model.align_y_to_x(zgy).cpu())
            out['shared_gpi_aligned'].append(model.align_x_to_y(zgx).cpu())

    for k in out:
        out[k] = torch.cat(out[k], dim=0)
    return out

# (optional but robust) ensure numpy arrays are CPU + float and finite
def to_np(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
    x = np.asarray(x, dtype=np.float32)
    # guard against NaNs/Infs that can break UMAP/PCA
    if not np.all(np.isfinite(x)):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x

def flatten_latents(latents):
    N, T, D = latents.shape
    return latents.reshape(N*T, D)

#______Variance evaluation functions used for dimension seep evaluation
def _fve(y_true, y_pred):
    """
    Fraction of Variance Explained (multi-output R^2 using global mean).
    y_*: (S, C)
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2)
    # Guard division-by-zero
    if ss_tot <= 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)

def cv_decode_fve(Z, Y, alphas=(1e-3, 1e-2, 1e-1, 1, 10), k=5, seed=0):
    """
    Cross-validated FVE with ridge regression and alpha selection inside CV.
    Z: (S, D)
    Y: (S, C)
    Returns: mean CV FVE
    """
    if Z.ndim != 2 or Y.ndim != 2:
        raise ValueError(f"cv_decode_fve expects 2D arrays, got {Z.shape=}, {Y.shape=}")
    S = Z.shape[0]
    if S < k:
        # Reduce folds if too few samples after lagging
        k = max(2, min(3, S))
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)

    best_alpha, best_cv = None, -np.inf
    for a in alphas:
        scores = []
        for tr, te in kf.split(Z):
            Ztr, Zte = Z[tr], Z[te]
            Ytr, Yte = Y[tr], Y[te]

            zsc = StandardScaler().fit(Ztr)
            ysc = StandardScaler().fit(Ytr)
            Ztr_s, Zte_s = zsc.transform(Ztr), zsc.transform(Zte)
            Ytr_s = ysc.transform(Ytr)

            W = Ridge(alpha=a, fit_intercept=False).fit(Ztr_s, Ytr_s)
            Yhat_s = W.predict(Zte_s)
            Yhat = ysc.inverse_transform(Yhat_s)

            scores.append(_fve(Yte, Yhat))
        mean_cv = float(np.mean(scores)) if scores else -np.inf
        if mean_cv > best_cv:
            best_cv, best_alpha = mean_cv, a

    # Final CV reporting with best alpha
    scores = []
    for tr, te in kf.split(Z):
        Ztr, Zte = Z[tr], Z[te]
        Ytr, Yte = Y[tr], Y[te]

        zsc = StandardScaler().fit(Ztr)
        ysc = StandardScaler().fit(Ytr)
        Ztr_s, Zte_s = zsc.transform(Ztr), zsc.transform(Zte)
        Ytr_s = ysc.transform(Ytr)

        W = Ridge(alpha=best_alpha, fit_intercept=False).fit(Ztr_s, Ytr_s)
        Yhat_s = W.predict(Zte_s)
        Yhat = ysc.inverse_transform(Yhat_s)

        scores.append(_fve(Yte, Yhat))
    return float(np.mean(scores)) if scores else 0.0

def safe_clip(x, lo=0.0):
    return float(max(lo, x))

def _cca_topk(X, Y, k=3, eps=1e-6, ridge=None):
    """
    Robust CCA for validation:
      • centers X,Y
      • adds scale-aware ridge to Cxx,Cyy
      • uses eigh for whitening and SVD for canonical correlations
    X, Y can be (B,T,D) or (N,D). Returns {"mean": float, "vals": 1D tensor}.
    """
    with torch.no_grad():
        # Flatten any batch/time dims to N
        if X.dim() == 3: X = X.reshape(-1, X.shape[-1])
        if Y.dim() == 3: Y = Y.reshape(-1, Y.shape[-1])

        # Early outs if not enough samples
        N, Dx = X.shape[0], X.shape[1]
        Dy = Y.shape[1]
        if N < 3 or Dx == 0 or Dy == 0:
            return {"mean": 0.0, "vals": torch.zeros(0)}

        # Center
        Xc = X - X.mean(0, keepdim=True)
        Yc = Y - Y.mean(0, keepdim=True)

        denom = max(1, N - 1)
        Cxx = (Xc.T @ Xc) / denom
        Cyy = (Yc.T @ Yc) / denom
        Cxy = (Xc.T @ Yc) / denom

        # Symmetrize for safety
        Cxx = 0.5 * (Cxx + Cxx.T)
        Cyy = 0.5 * (Cyy + Cyy.T)

        # Scale-aware ridge: λ = α * trace(C)/D
        def add_ridge(C, explicit=None, alpha=1e-3):
            d = C.shape[0]
            lam = explicit
            if lam is None:
                lam = float((torch.trace(C) / max(1, d)) * alpha)
            return C + lam * torch.eye(d, device=C.device, dtype=C.dtype)

        Cxx_r = add_ridge(Cxx, ridge)
        Cyy_r = add_ridge(Cyy, ridge)

        # Whitening via eigh (safe with clamp_min)
        Exx, Uxx = torch.linalg.eigh(Cxx_r)
        Eyy, Uyy = torch.linalg.eigh(Cyy_r)
        Wx = Uxx @ torch.diag(Exx.clamp_min(eps).rsqrt()) @ Uxx.T
        Wy = Uyy @ torch.diag(Eyy.clamp_min(eps).rsqrt()) @ Uyy.T

        # Correlation matrix in whitened space
        M = Wx @ Cxy @ Wy

        # Canonical correlations via SVD (most stable)
        s = torch.linalg.svdvals(M)
        s_sorted = torch.sort(s, descending=True).values
        # topk = s_sorted[:min(k, s_sorted.numel())]
        # return {"mean": float(topk.mean().item()) if topk.numel() > 0 else 0.0,
        #         "vals": topk.cpu()}
        
        # --- robust k handling ---
        if k is None:
            k = s_sorted.numel()                 # use all
        # clamp & cast to int
        k = int(max(0, min(int(k), s_sorted.numel())))

        topk = s_sorted[:k]
        return {
            "mean": float(topk.mean().item()) if k > 0 else 0.0,
            "vals": topk.detach().cpu()
        }

def _safe_cca_topk(a, b, k=None):
    """
    a,b are (B,T,D) -> we reshape to (BT, D) like in train.
    """
    try:
        # flatten to (BT, D)
        A = a.reshape(-1, a.shape[-1])
        B = b.reshape(-1, b.shape[-1])
        res = _cca_topk(A, B, k=k)  # expects tensors, returns dict with "mean"
        return float(res["mean"])
    except NameError:
        return None

@torch.no_grad()
def get_measuresAll_df_per_sample(
    model, gpi_test, stn_test, device, side, subject_id=None, include_alignment=True
):
    """
    Returns two DataFrames:
      - df_recon: MSE/R2 for full/self/cross decodes (raw + aligned cross if requested)
      - df_sim:   similarity metrics (cosine & CCA when available)
    """
    model.eval()
    recon_records = []
    sim_records   = []

    N = gpi_test.shape[0]
    for i in range(N):
        xb = gpi_test[i:i+1].to(device)  # (1, T, Cx)
        yb = stn_test[i:i+1].to(device)  # (1, T, Cy)

        # forward once
        recon_xb, recon_yb, shared_xb, shared_yb, private_xb, private_yb = model(xb, yb, private_gate=1.0)

        # --- aligned shared latents (for cross decodes & shared-shared similarity) ---
        if include_alignment:
            # uses model.use_identity_align to decide conv+mapper vs mapper-only
            shared_yb_to_x = model.align_y_to_x(shared_yb)  # y->x aligned to x space
            shared_xb_to_y = model.align_x_to_y(shared_xb)  # x->y aligned to y space
        else:
            shared_yb_to_x = shared_yb
            shared_xb_to_y = shared_xb

        # --- build decode variants (avoid redundant decodes) ---
        # self shared-only decodes
        x_self_shared = model.decoder_gpi(shared_xb, torch.zeros_like(private_xb))
        y_self_shared = model.decoder_stn(shared_yb, torch.zeros_like(private_yb))
        # private-only decodes
        x_priv_only   = model.decoder_gpi(torch.zeros_like(shared_xb), private_xb)
        y_priv_only   = model.decoder_stn(torch.zeros_like(shared_yb), private_yb)
        # cross using raw shared (un-aligned)
        x_from_y_raw  = model.decoder_gpi(shared_yb, torch.zeros_like(private_yb))
        y_from_x_raw  = model.decoder_stn(shared_xb, torch.zeros_like(private_xb))
        # cross using aligned shared (preferred for evaluation)
        x_from_y_aln  = model.decoder_gpi(shared_yb_to_x, torch.zeros_like(private_yb))
        y_from_x_aln  = model.decoder_stn(shared_xb_to_y, torch.zeros_like(private_xb))

        outputs = {
            "gpi_full": recon_xb,
            "gpi_private": x_priv_only,
            "gpi_shared_gpi": x_self_shared,
            "gpi_shared_stn_raw": x_from_y_raw,
            "gpi_shared_stn_aligned": x_from_y_aln,

            "stn_full": recon_yb,
            "stn_private": y_priv_only,
            "stn_shared_stn": y_self_shared,
            "stn_shared_gpi_raw": y_from_x_raw,
            "stn_shared_gpi_aligned": y_from_x_aln,
        }
        targets = {"gpi": xb, "stn": yb}

        # --- MSE & R² per decode condition ---
        for key, out in outputs.items():
            region = key.split("_")[0]  # "gpi" or "stn"
            tgt = targets[region]
            mse = F.mse_loss(out, tgt, reduction='mean').item()

            # R2 over all time×channels
            tgt_np = tgt.detach().cpu().numpy().astype(np.float32)
            r2 = r2_score(tgt_np.ravel(), tgt_np.ravel())

            recon_records.append({
                "subject": subject_id, "side": side, "sample": i,
                "condition": key, "mse": mse, "r2": r2
            })

        # ---------- Similarity metrics ----------
        # 1) Shared↔Shared similarity (raw & aligned)
        cca_shared_raw = _safe_cca_topk(shared_xb, shared_yb, k=None)
        cca_shared_aln = _safe_cca_topk(shared_xb, shared_yb_to_x, k=None)

        if cca_shared_raw is not None:
            sim_records.append({
                "subject": subject_id, "side": side, "sample": i,
                "pair": "shared_x ~ shared_y (raw)", "metric": "cca_mean", "value": cca_shared_raw
            })

        if cca_shared_aln is not None:
            sim_records.append({
                "subject": subject_id, "side": side, "sample": i,
                "pair": "shared_x ~ shared_y (aligned y->x)", "metric": "cca_mean", "value": cca_shared_aln
            })

        # 2) Leakage (shared↔private within region; want low)
        cca_leak_x = _safe_cca_topk(shared_xb, private_xb, k=None)
        cca_leak_y = _safe_cca_topk(shared_yb, private_yb, k=None)

        if cca_leak_x is not None:
            sim_records.append({"subject": subject_id, "side": side, "sample": i,
                                "pair": "shared_x ~ private_x", "metric": "cca_mean", "value": cca_leak_x})

        if cca_leak_y is not None:
            sim_records.append({"subject": subject_id, "side": side, "sample": i,
                                "pair": "shared_y ~ private_y", "metric": "cca_mean", "value": cca_leak_y})

        # 3) Private↔Private across regions (should be dissimilar/independent)
        cca_priv_priv = _safe_cca_topk(private_xb, private_yb, k=None)

        if cca_priv_priv is not None:
            sim_records.append({"subject": subject_id, "side": side, "sample": i,
                                "pair": "private_x ~ private_y", "metric": "cca_mean", "value": cca_priv_priv})

    df_recon = pd.DataFrame(recon_records)
    df_sim   = pd.DataFrame(sim_records)
    return df_recon, df_sim


#onstim latents
def extract_latents_by_condition(model, X_test, Y_test, labels, device, label_map,include_alignment = True):
    """
    Extracts shared and private latents for GPi and STN from test data, grouped by stimulation condition.

    Returns:
        shared_gpi_dict:    dict of condition -> (N, latent_dim) tensor
        shared_stn_dict:    dict of condition -> (N, latent_dim) tensor
        private_gpi_dict:   dict of condition -> (N, latent_dim) tensor
        private_stn_dict:   dict of condition -> (N, latent_dim) tensor
    """
    model.eval()

    shared_gpi_dict = defaultdict(list)
    shared_stn_dict = defaultdict(list)
    private_gpi_dict = defaultdict(list)
    private_stn_dict = defaultdict(list)
    shared_gpi_aligned_dict = defaultdict(list)
    shared_stn_aligned_dict = defaultdict(list)

    with torch.no_grad():
        for xb, yb, label in zip(X_test, Y_test, labels):
            xb = xb.unsqueeze(0).to(device)
            yb = yb.unsqueeze(0).to(device)
            label_str = label_map[label.item()]

            _, _, shared_gpi, shared_stn, private_gpi, private_stn = model(xb, yb)

            shared_gpi_dict[label_str].append(shared_gpi.squeeze(0).cpu())
            shared_stn_dict[label_str].append(shared_stn.squeeze(0).cpu())
            private_gpi_dict[label_str].append(private_gpi.squeeze(0).cpu())
            private_stn_dict[label_str].append(private_stn.squeeze(0).cpu())
                # --- aligned shared latents (for cross decodes & shared-shared similarity) ---
            if include_alignment:
                # uses model.use_identity_align to decide conv+mapper vs mapper-only
                shared_yb_to_x = model.align_y_to_x(shared_stn)  # y->x aligned to x space
                shared_xb_to_y = model.align_x_to_y(shared_gpi)  # x->y aligned to y space
            else:
                shared_yb_to_x = shared_stn
                shared_xb_to_y = shared_gpi

            shared_gpi_aligned_dict[label_str].append(shared_yb_to_x.squeeze(0).cpu())
            shared_stn_aligned_dict[label_str].append(shared_xb_to_y.squeeze(0).cpu())

    # Convert lists to stacked tensors
    for d in [shared_gpi_dict, shared_stn_dict, private_gpi_dict, private_stn_dict,shared_gpi_aligned_dict,shared_stn_aligned_dict]:
        for key in d:
            d[key] = torch.stack(d[key])

    return shared_gpi_dict, shared_stn_dict, private_gpi_dict, private_stn_dict,shared_gpi_aligned_dict,shared_stn_aligned_dict


# onstim evaluations:
def _make_samples(x, sample_level="timepoints"):
    """
    x: numpy array of shape (N,T,D) or (M,D)
    sample_level:
      - "timepoints": returns (N*T, D) by flattening across time
      - "windows":    returns (N, D) by averaging over time per window
    """
    x = np.asarray(x)
    if x.ndim == 3:
        N, T, D = x.shape
        if sample_level == "timepoints":
            X = x.reshape(N*T, D)
        elif sample_level == "windows":
            X = x.mean(axis=1)  # (N,D)
        else:
            raise ValueError("sample_level must be {'timepoints','windows'}")
    elif x.ndim == 2:
        X = x
    else:
        raise ValueError(f"Expected (N,T,D) or (M,D); got {x.shape}")
    return X

def _subsample_rows(X, max_n=2000, rng=None):
    """Optionally sub-sample rows for O(n^2) metrics."""
    if max_n is None or X.shape[0] <= max_n:
        return X
    rng = np.random.default_rng(None if rng is None else rng)
    idx = rng.choice(X.shape[0], size=max_n, replace=False)
    return X[idx]
# -------------------------
# Energy Distance
# -------------------------
def energy_distance(X, Y, max_n=2000, rng=None):
    """
    Székely–Rizzo energy distance using Euclidean norm.
    Returns a scalar.
    """
    X = _subsample_rows(np.asarray(X), max_n, rng)
    Y = _subsample_rows(np.asarray(Y), max_n, rng)

    d_xy = pairwise_distances(X, Y, metric="euclidean").mean()
    d_xx = pairwise_distances(X, X, metric="euclidean").mean()
    d_yy = pairwise_distances(Y, Y, metric="euclidean").mean()
    return 2.0 * d_xy - d_xx - d_yy

# -------------------------
# MMD (RBF, unbiased)
# -------------------------
def _median_heuristic_bandwidth(X, Y, max_n=2000, rng=None):
    XY = np.vstack([
        _subsample_rows(np.asarray(X), max_n, rng),
        _subsample_rows(np.asarray(Y), max_n, rng),
    ])
    # median of pairwise distances; gamma = 1/(2*sigma^2)
    D = pairwise_distances(XY, XY, metric="euclidean")
    # use upper triangle without diagonal
    tri = D[np.triu_indices_from(D, k=1)]
    sigma = np.median(tri[tri > 0]) + 1e-12
    return sigma

def mmd_unbiased_rbf(X, Y, bandwidth=None, max_n=2000, rng=None):
    """
    Unbiased MMD^2 with RBF kernel k(u,v)=exp(-||u-v||^2/(2*sigma^2)).
    Returns MMD (not squared) for interpretability.
    """
    X = _subsample_rows(np.asarray(X), max_n, rng)
    Y = _subsample_rows(np.asarray(Y), max_n, rng)

    if bandwidth is None:
        sigma = _median_heuristic_bandwidth(X, Y, max_n=1000, rng=rng)
    else:
        sigma = float(bandwidth)
    inv_two_sigma2 = 1.0 / (2.0 * sigma * sigma)

    # pairwise squared distances
    XX = pairwise_distances(X, X, metric="sqeuclidean")
    YY = pairwise_distances(Y, Y, metric="sqeuclidean")
    XY = pairwise_distances(X, Y, metric="sqeuclidean")

    # unbiased estimator: exclude diagonals for XX, YY
    n = X.shape[0]
    m = Y.shape[0]
    np.fill_diagonal(XX, np.nan)
    np.fill_diagonal(YY, np.nan)

    Kxx = np.exp(-inv_two_sigma2 * XX)
    Kyy = np.exp(-inv_two_sigma2 * YY)
    Kxy = np.exp(-inv_two_sigma2 * XY)

    term_xx = np.nanmean(Kxx)  # average over i!=j
    term_yy = np.nanmean(Kyy)
    term_xy = Kxy.mean()

    mmd2 = term_xx + term_yy - 2.0 * term_xy
    return float(np.sqrt(max(mmd2, 0.0)))  # report MMD (sqrt of MMD^2)

# -------------------------
# Main: compute per-latent, per-frequency (+ Off baseline)
# -------------------------
def distribution_metrics_with_baseline(
    shared_gpi, shared_stn, private_gpi, private_stn,
    side, setting, subject_id,
    metrics=("energy", "mmd"),
    sample_level="timepoints",      # or "windows"
    max_n=2000,
    rng_seed=0,
    baseline="holdout_mean",        # {'holdout_mean', None}
    holdout_frac=0.5,               # fraction of Off rows to use in holdout mean
    mmd_bandwidth=None              # None => median heuristic
):
    """
    Returns a tidy DataFrame with columns:
    [subject, side, setting, frequency, latent_type, metric, value]
    """
    all_latents = {
        "shared_gpi": shared_gpi,
        "shared_stn": shared_stn,
        "private_gpi": private_gpi,
        "private_stn": private_stn,
    }
    rows = []
    rng = np.random.default_rng(rng_seed)

    for latent_type, dct in all_latents.items():
        if "Off" not in dct:
            continue

        # Build sample matrices
        off_np=dct["Off"].detach().cpu().numpy().astype(np.float32)
        off = _make_samples(off_np, sample_level)  # (M_off, D)

        # ---------- Off baseline (optional) ----------
        if baseline == "holdout_mean":
            M = off.shape[0]
            k = max(int(M * holdout_frac), 50)
            idx = rng.choice(M, size=min(k, M), replace=False)
            off_hold = off[idx]

            # For distributional baseline we compare Off_hold vs Off_all
            baseline_pairs = [("OffBaseline", off_hold, off)]
            # Note: this is apples-to-apples with Off vs On distribution comparisons
        else:
            baseline_pairs = []

        # ---------- Frequencies present ----------
        freq_keys = [k for k in dct.keys() if k != "Off"]
        for fk in freq_keys:
            on_np = dct[fk].detach().cpu().numpy().astype(np.float32)
            on = _make_samples(on_np, sample_level)
            baseline_pairs.append((fk, on, off))

        # ---------- Compute metrics ----------
        for freq, A, B in baseline_pairs:
            for mname in metrics:
                if mname == "energy":
                    val = energy_distance(A, B, max_n=max_n, rng=rng)
                elif mname == "mmd":
                    val = mmd_unbiased_rbf(A, B, bandwidth=mmd_bandwidth, max_n=max_n, rng=rng)
                else:
                    raise ValueError(f"Unknown metric: {mname}")

                rows.append({
                    "subject": subject_id,
                    "side": side,
                    "setting": setting,
                    "frequency": freq,
                    "latent_type": latent_type,
                    "metric": mname,
                    "value": float(val),
                    "sample_level": sample_level,
                    "max_n": max_n
                })

    return pd.DataFrame(rows)


#classification:
def _prepare_latents_without_averaging(latent_dict):
    """
    Convert (N, T, D) latent tensors into (N*T, D) feature vectors,
    using each timepoint as a sample.
    """
    data = []
    labels = []
    condition_map = {"Off": 0, "85Hz": 1, "185Hz": 2, "250Hz": 3}

    for condition, tensor in latent_dict.items():
        N, T, D = tensor.shape
        flat = tensor.reshape(N * T, D)  # shape: (N*T, D)
        data.append(flat)
        labels.extend([condition_map[condition]] * (N * T))

    X = np.vstack(data)
    y = np.array(labels)
    return X, y

def calculate_RF_accuracy(shared_gpi, shared_stn, private_gpi, private_stn, side, setting, subject_id):
    latent_sets = {
        "shared_gpi": shared_gpi,
        "shared_stn": shared_stn,
        "private_gpi": private_gpi,
        "private_stn": private_stn,
    }

    results_accuracy = []
    results_importance = []

    for latent_type, latent_dict in latent_sets.items():
        X, y = _prepare_latents_without_averaging(latent_dict)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=0.2, random_state=42
        )

        # Normalize
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        # Save classification accuracy
        results_accuracy.append({
            "subject": subject_id,
            "side": side,
            "setting": setting,
            "latent_type": latent_type,
            "accuracy": acc,
            "precision_macro": precision_score(y_test, y_pred, average='macro'),
            "recall_macro": recall_score(y_test, y_pred, average='macro'),
            "f1_macro": f1_score(y_test, y_pred, average='macro'),
            "n_samples": len(y)
        })

        # Save feature importances
        for i, imp in enumerate(clf.feature_importances_):
            results_importance.append({
                "subject": subject_id,
                "side": side,
                "setting": setting,
                "latent_type": latent_type,
                "feature_dim": i,
                "importance": imp,
            })

    return pd.DataFrame(results_accuracy), pd.DataFrame(results_importance)