
import torch
from sklearn.cross_decomposition import CCA
from scipy.linalg import subspace_angles
import pandas as pd
import numpy as np
import os

from data.data_loader import build_dataset_with_lag

def _to_numpy(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
# -----------------------------
# Latent extraction wrapper
# -----------------------------
def extract_latents_from_test_set(model, gpi_test, stn_test, device):
    """
    Extract shared and private latents from test tensors.
    
    Args:
        model: trained model
        gpi_test, stn_test: torch tensors of shape (N, T, C)
        device: CUDA or CPU

    Returns:
        shared_gpi_all, shared_stn_all,
        private_gpi_all, private_stn_all: all shape (N, T, D)
    """
    model.eval()
    shared_gpi_all, shared_stn_all = [], []
    private_gpi_all, private_stn_all = [], []

    with torch.no_grad():
        for xb, yb in zip(gpi_test, stn_test):
            xb = xb.unsqueeze(0).to(device)
            yb = yb.unsqueeze(0).to(device)

            _, _, shared_gpi, shared_stn, private_gpi, private_stn = model(xb, yb)

            shared_gpi_all.append(shared_gpi.squeeze(0).cpu())
            shared_stn_all.append(shared_stn.squeeze(0).cpu())
            private_gpi_all.append(private_gpi.squeeze(0).cpu())
            private_stn_all.append(private_stn.squeeze(0).cpu())

    return (
        torch.stack(shared_gpi_all),
        torch.stack(shared_stn_all),
        torch.stack(private_gpi_all),
        torch.stack(private_stn_all),
    )

def _extract_latents_for_eval(model, reg1_tensor, reg2_tensor, device):
    """
    Uses existing helper to get latents on the evaluation tensors and makes them numpy
    Assumes extract_latents_from_test_set(model, x1, x2, device) is defined.
    """
    model.eval()
    with torch.no_grad():
        shared_reg1, shared_reg2, private_reg1, private_reg2 = extract_latents_from_test_set(
            model, reg1_tensor, reg2_tensor, device
        )
    return tuple(_to_numpy(z) for z in (shared_reg1, shared_reg2, private_reg1, private_reg2))

def compare_latent_performance(model_latents, gt_latents, model_name="Model"):
    """
    Compare learned latents to GT using CCA, subspace angle, and per-dim MSE.

    Parameters
    ----------
    model_latents : np.ndarray or torch.Tensor, shape (n_trials, T_model, D)
    gt_latents    : np.ndarray or torch.Tensor, shape (n_trials, D, T_gt)
    model_name    : str

    Returns
    -------
    results_df : pd.DataFrame  (per-dim CCA, angle, MSE)
    metrics    : dict           (mean_cca, mean_angle, mean_mse)
    """
    # Ensure numpy
    if hasattr(model_latents, "detach"):
        model_latents = model_latents.detach().cpu().numpy()
    if hasattr(gt_latents, "detach"):
        gt_latents = gt_latents.detach().cpu().numpy()

    n_trials, T_model, D = model_latents.shape

    # GT is (N, D, T_gt) -> (N, T_gt, D); then trim to match model T
    gt_latents = np.transpose(gt_latents, (0, 2, 1))  # (N, T_gt, D)
    gt_latents_trimmed = gt_latents[:, -T_model:, :]  # (N, T_model, D)

    # Flatten across trials/time -> (N*T, D)
    model_flat = model_latents.reshape(-1, D)
    gt_flat = gt_latents_trimmed.reshape(-1, D)

    # CCA (D comps)
    cca = CCA(n_components=D)
    model_cca, gt_cca = cca.fit_transform(model_flat, gt_flat)
    cca_corrs = [np.corrcoef(model_cca[:, i], gt_cca[:, i])[0, 1] for i in range(D)]
    mean_cca = float(np.mean(cca_corrs))

    # Subspace angles (principal angles between rank-D subspaces)
    # Build orthonormal bases via SVD of zero-mean data
    m_center = model_flat - model_flat.mean(axis=0, keepdims=True)
    g_center = gt_flat   - gt_flat.mean(axis=0, keepdims=True)
    _, _, VT_m = np.linalg.svd(m_center, full_matrices=False)
    _, _, VT_g = np.linalg.svd(g_center, full_matrices=False)
    Bm = VT_m[:D].T
    Bg = VT_g[:D].T
    angles_deg = np.degrees(subspace_angles(Bm, Bg))
    mean_angle = float(np.mean(angles_deg))

    # Per-dim MSE
    mse_per_dim = np.mean((model_flat - gt_flat) ** 2, axis=0)
    mean_mse = float(np.mean(mse_per_dim))

    results_df = pd.DataFrame({
        "CCA Correlation": cca_corrs,
        "Subspace Angle (deg)": angles_deg,
        "MSE": mse_per_dim
    })

    metrics = {
        "mean_cca": mean_cca,
        "mean_angle": mean_angle,
        "mean_mse": mean_mse
    }
    return results_df, metrics

def mean_cca(a: np.ndarray, b: np.ndarray, n_comp: int):
    """
    a,b: (N,T,D) -> flattened to (N*T, D). Returns mean CCA over D comps.
    """
    A = a.reshape(-1, a.shape[-1])
    B = b.reshape(-1, b.shape[-1])
    cca = CCA(n_components=n_comp)
    u, v = cca.fit_transform(A, B)
    corrs = [np.corrcoef(u[:, i], v[:, i])[0, 1] for i in range(n_comp)]
    return float(np.mean(corrs)), corrs

# ---------- Generic latent alignment helper ----------

@torch.no_grad()
def _align_latent(model, z: torch.Tensor, *, src: str, dst: str) -> torch.Tensor:
    """
    Align latent z from `src` region space to `dst` region space.

    Works with either:
      - a composite wrapper (model.align_y2x / model.align_x2y), OR
      - individual pieces (conv_*2*, map_*2*, shift_*).

    src, dst in {"x","y"}.
    Returns a tensor on the same device/dtype as z.
    """
    assert src in {"x", "y"} and dst in {"x", "y"}
    if src == dst:
        return z

    # Prefer a single composite aligner if provided.
    attr = f"align_{src}2{dst}"
    fn = getattr(model, attr, None)
    if callable(fn):
        return fn(z)

    # Fallback: compose known pieces in a sensible order.
    # We do conv -> map -> shift (broad → specific → fine).
    pieces = []
    if src == "y" and dst == "x":
        pieces = ["conv_y2x", "map_y2x", "shift_y"]
    elif src == "x" and dst == "y":
        pieces = ["conv_x2y", "map_x2y", "shift_x"]

    out = z
    for name in pieces:
        fn = getattr(model, name, None)
        if callable(fn):
            out = fn(out)
    return out

# ---------- model adapters ----------
def _get_decoder_r1(model):
    for name in ["decoder_gpi", "decoder_r1", "decoder_x", "decoder1", "decoder_region1"]:
        if hasattr(model, name):
            return getattr(model, name)
    raise AttributeError("Could not find decoder for region1 on model.")

def _get_decoder_r2(model):
    for name in ["decoder_stn", "decoder_r2", "decoder_y", "decoder2", "decoder_region2"]:
        if hasattr(model, name):
            return getattr(model, name)
    raise AttributeError("Could not find decoder for region2 on model.")

def _r2_per_channel(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9):
    """
    y_*: (N, T, C) or (N*T, C). Returns mean R² over channels and trials/time.
    """
    if y_true.ndim == 3:
        N, T, C = y_true.shape
        yt = y_true.reshape(N*T, C)
        yp = y_pred.reshape(N*T, C)
    else:
        yt, yp = y_true, y_pred
        C = yt.shape[-1]
    # Center w.r.t. mean of y_true (per-channel)
    yt_c = yt - yt.mean(axis=0, keepdims=True)
    ss_tot = np.sum(yt_c**2, axis=0) + eps
    ss_res = np.sum((yt - yp)**2, axis=0)
    r2 = 1.0 - (ss_res / ss_tot)
    return float(np.mean(r2)), r2  # (mean over channels, vector per channel)

def _add_derived_columns(run_agg: pd.DataFrame) -> pd.DataFrame:
    # Cross vs shared-only consistency (smaller is better)
    d1 = (run_agg["r2_y_shared_only"] - run_agg["r2_y_from_x_shared"]).abs()
    d2 = (run_agg["r2_x_shared_only"] - run_agg["r2_x_from_y_shared"]).abs()
    run_agg["r2_shared_consistency"] = 0.5*(d1 + d2)

    # Leakage normalized by signal
    run_agg["leakage_ratio"] = (
        np.maximum(run_agg["leak_shared_vs_gt_private"], run_agg["leak_private_vs_gt_shared"]) /
        run_agg["mean_cca_shared"].clip(lower=1e-6)
    )

    # A simple, transparent Stage-1 ranking score
    run_agg["score_stage1"] = (
        0.45*run_agg["mean_cca_shared"] +
        0.20*run_agg["cca_shared12"] +
        0.20*((run_agg["r2_y_from_x_shared"] + run_agg["r2_x_from_y_shared"])/2.0) +
        0.10*((run_agg["r2_y_shared_only"] + run_agg["r2_x_shared_only"])/2.0) -
        0.05*run_agg["r2_shared_consistency"] -    # penalize big gaps
        0.05*run_agg["leakage_ratio"]               # penalize relative leakage
    )
    return run_agg
# -----------------------------
# One model × one dataset
# -----------------------------
@torch.no_grad()
def evaluate_one_model_on_one_dataset(model, data, lags, device, run_name, latent_save_dir):
    """
    1) rebuild lagged inputs
    2) extract latents
    3) save latents
    4) metrics vs GT (CCA/angles/MSE) + extra diagnostics:
       - Reconstruction R² per region (full / shared-only / private-only)
       - Cross-prediction R² (Y from shared-X only; X from shared-Y only)
       - Leakage CCAs (shared vs GT-private; private vs GT-shared)
       - Shared alignment (CCA between learned shared1 and shifted shared2)
    """
    # -------- build inputs exactly like training --------
    reg1_R, reg2_R = build_dataset_with_lag(data["region1"], data["region2"], lags=lags)
    reg1_tensor = torch.tensor(reg1_R, dtype=torch.float32).permute(0, 2, 1).to(device)  # (N,T,C)
    reg2_tensor = torch.tensor(reg2_R, dtype=torch.float32).permute(0, 2, 1).to(device)

    model = model.to(device).eval()

    # -------- extract latents (N,T,D) --------
    shared_r1, shared_r2, private_r1, private_r2 = _extract_latents_for_eval(
        model, reg1_tensor, reg2_tensor, device
    )
    # keep numpy copies for some metrics
    s1_np = _to_numpy(shared_r1); s2_np = _to_numpy(shared_r2)
    p1_np = _to_numpy(private_r1); p2_np = _to_numpy(private_r2)

    # tensors for decoding
    s1_t = torch.tensor(s1_np, device=device, dtype=torch.float32)
    p1_t = torch.tensor(p1_np, device=device, dtype=torch.float32)
    s2_t = torch.tensor(s2_np, device=device, dtype=torch.float32)
    p2_t = torch.tensor(p2_np, device=device, dtype=torch.float32)

    # -------- save latents --------
    os.makedirs(latent_save_dir, exist_ok=True)
    latent_npz_path = os.path.join(latent_save_dir, f"{run_name}_latents.npz")
    np.savez_compressed(latent_npz_path,
        shared_reg1=s1_np, shared_reg2=s2_np,
        private_reg1=p1_np, private_reg2=p2_np
    )

    # -------- GT comparisons (existing) --------
    gt_s1 = data["gt_shared1"]; gt_s2 = data["gt_shared2"]
    gt_p1 = data["gt_private1"]; gt_p2 = data["gt_private2"]

    _, m_shared1 = compare_latent_performance(s1_np, gt_s1, model_name=f"{run_name}_Shared1")
    _, m_shared2 = compare_latent_performance(s2_np, gt_s2, model_name=f"{run_name}_Shared2")
    _, m_priv1   = compare_latent_performance(p1_np, gt_p1, model_name=f"{run_name}_Private1")
    _, m_priv2   = compare_latent_performance(p2_np, gt_p2, model_name=f"{run_name}_Private2")

    # -------- extra: leakage checks (shared↔private GT) --------
    Tm = s1_np.shape[1]  # T_model
    gt_p1_T = np.transpose(gt_p1, (0,2,1))[:, -Tm:, :]
    gt_s1_T = np.transpose(gt_s1, (0,2,1))[:, -Tm:, :]
    gt_p2_T = np.transpose(gt_p2, (0,2,1))[:, -Tm:, :]
    gt_s2_T = np.transpose(gt_s2, (0,2,1))[:, -Tm:, :]

    mean_cca_s1_vs_gtp1, _ = mean_cca(s1_np, gt_p1_T, n_comp=s1_np.shape[-1])
    mean_cca_p1_vs_gts1, _ = mean_cca(p1_np, gt_s1_T, n_comp=p1_np.shape[-1])
    mean_cca_s2_vs_gtp2, _ = mean_cca(s2_np, gt_p2_T, n_comp=s2_np.shape[-1])
    mean_cca_p2_vs_gts2, _ = mean_cca(p2_np, gt_s2_T, n_comp=p2_np.shape[-1])

    # -------- extra: shared alignment across regions --------
    # use shift layers if present for cross-region lag alignment
    # s1_align = _maybe_shift(model, "x", torch.tensor(s1_np, device=device, dtype=torch.float32)).cpu().numpy()
    # s2_align = _maybe_shift(model, "y", torch.tensor(s2_np, device=device, dtype=torch.float32)).cpu().numpy()
    # mean_cca_shared12, _ = mean_cca(s1_align, s2_align, n_comp=s1_np.shape[-1])

    # Align ONE side into the other's space (here: y → x)
    s2_aligned = _align_latent(model, s2_t, src="y", dst="x")

    mean_cca_shared12, _ = mean_cca(
        s1_t.detach().cpu().numpy(),
        s2_aligned.detach().cpu().numpy(),
        n_comp=s1_t.shape[-1]
    )

    # -------- extra: reconstruction & cross-prediction R² --------
    # decoders
    dec_r1 = _get_decoder_r1(model)
    dec_r2 = _get_decoder_r2(model)

    zeros_p1 = torch.zeros_like(p1_t); zeros_p2 = torch.zeros_like(p2_t)
    zeros_s1 = torch.zeros_like(s1_t); zeros_s2 = torch.zeros_like(s2_t)

    # full reconstruction
    x_full = dec_r1(s1_t, p1_t)
    y_full = dec_r2(s2_t, p2_t)

    # shared-only / private-only (same-region decoders)
    x_shared_only = dec_r1(s1_t, zeros_p1)
    x_private_only = dec_r1(zeros_s1, p1_t)
    y_shared_only = dec_r2(s2_t, zeros_p2)
    y_private_only = dec_r2(zeros_s2, p2_t)

    @torch.no_grad()
    def _decode_shared(decoder, z_shared, zeros_private):
        # Prefer decoder.decode_shared if it exists; otherwise pass zeros for private
        fn = getattr(decoder, "decode_shared", None)
        return fn(z_shared) if callable(fn) else decoder(z_shared, zeros_private)

        # map shared-X → Y-space (so it matches decoder_r2’s shared coordinates)
    s1_in_y = _align_latent(model, s1_t, src="x", dst="y")
    # map shared-Y → X-space (so it matches decoder_r1’s shared coordinates)
    s2_in_x = _align_latent(model, s2_t, src="y", dst="x")

    y_from_x_shared = _decode_shared(model.decoder_stn, s1_in_y, zeros_p2)  # region2 from X's shared
    x_from_y_shared = _decode_shared(model.decoder_gpi, s2_in_x, zeros_p1)  # region1 from Y's shared

    # convert to numpy
    x_true = reg1_tensor.detach().cpu().numpy()
    y_true = reg2_tensor.detach().cpu().numpy()
    x_full_np = x_full.detach().cpu().numpy()
    y_full_np = y_full.detach().cpu().numpy()
    x_sh_np = x_shared_only.detach().cpu().numpy()
    x_pr_np = x_private_only.detach().cpu().numpy()
    y_sh_np = y_shared_only.detach().cpu().numpy()
    y_pr_np = y_private_only.detach().cpu().numpy()
    y_from_x_np = y_from_x_shared.detach().cpu().numpy()
    x_from_y_np = x_from_y_shared.detach().cpu().numpy()

    # R² metrics
    r2_x_full_mean,  _ = _r2_per_channel(x_true, x_full_np)
    r2_y_full_mean,  _ = _r2_per_channel(y_true, y_full_np)
    r2_x_sh_mean,    _ = _r2_per_channel(x_true, x_sh_np)
    r2_x_pr_mean,    _ = _r2_per_channel(x_true, x_pr_np)
    r2_y_sh_mean,    _ = _r2_per_channel(y_true, y_sh_np)
    r2_y_pr_mean,    _ = _r2_per_channel(y_true, y_pr_np)
    r2_y_from_x_mean,_ = _r2_per_channel(y_true, y_from_x_np)
    r2_x_from_y_mean,_ = _r2_per_channel(x_true, x_from_y_np)

    # -------- assemble tidy rows (repeat run-level extras on each latent row) --------
    common = {
        "latents_path": latent_npz_path,

        # leakage CCAs
        "cca_leak_s1_vs_gtp1": mean_cca_s1_vs_gtp1,
        "cca_leak_p1_vs_gts1": mean_cca_p1_vs_gts1,
        "cca_leak_s2_vs_gtp2": mean_cca_s2_vs_gtp2,
        "cca_leak_p2_vs_gts2": mean_cca_p2_vs_gts2,

        # shared alignment
        "cca_shared12": mean_cca_shared12,

        # recon & cross-pred R²
        "r2_x_full": r2_x_full_mean,
        "r2_y_full": r2_y_full_mean,
        "r2_x_shared_only": r2_x_sh_mean,
        "r2_x_private_only": r2_x_pr_mean,
        "r2_y_shared_only": r2_y_sh_mean,
        "r2_y_private_only": r2_y_pr_mean,
        "r2_y_from_x_shared": r2_y_from_x_mean,
        "r2_x_from_y_shared": r2_x_from_y_mean,
    }

    rows = [
        {"latent_type": "Shared1", **m_shared1, **common},
        {"latent_type": "Shared2", **m_shared2, **common},
        {"latent_type": "Private1", **m_priv1,   **common},
        {"latent_type": "Private2", **m_priv2,   **common},
    ]
    return rows

# -----------------------------
# Orchestrator over all models × datasets  (upgraded)
# -----------------------------
def evaluate_all_models_and_datasets(
    TRAINED_MODELS,
    ALL_DATASETS,
    save_excel_path,
    device="cuda" if torch.cuda.is_available() else "cpu",
    lags=3,
    latent_save_dir=r"F:\comp_project\synthecticData\latents_eval"
):
    """
    Loops over all trained runs and matching datasets (by regime),
    extracts & saves latents, evaluates metrics, and writes a multi-sheet Excel.
    """
    all_rows = []

    # fast lookup
    regime_to_data = {d["regime"]: d["data"] for d in ALL_DATASETS}

    for item in TRAINED_MODELS:
        regime  = item["regime"]
        variant = item["variant"]
        model   = item["model"]
        run_name = f"{regime}_{variant}"
        seed_model = item.get("seed", np.nan)  # default to NaN if missing

        if regime not in regime_to_data:
            print(f"[WARN] Regime {regime} not found in ALL_DATASETS; skipping.")
            continue

        data = regime_to_data[regime]

        try:
            rows = evaluate_one_model_on_one_dataset(
                model=model,
                data=data,
                lags=lags,
                device=device,
                run_name=run_name,
                latent_save_dir=latent_save_dir
            )
            for r in rows:
                r.update({"regime": regime, "variant": variant, "seed_label": seed_model})
            all_rows.extend(rows)
            print(f"[OK] {run_name}: evaluated and latents saved.")
        except Exception as e:
            print(f"[ERROR] {run_name}: {e}")
            continue

    if not all_rows:
        print("[WARN] No evaluation rows produced.")
        return pd.DataFrame()

    # ---------- tidy long table with ALL metrics ----------
    summary_df = pd.DataFrame(all_rows)

    # Put metadata first, keep ALL metric columns that exist
    meta_cols = ["regime", "variant","seed_label", "latent_type", "latents_path"]
    metric_cols = [c for c in summary_df.columns if c not in meta_cols]
    summary_df = summary_df[meta_cols + metric_cols]

    # ---------- run-level aggregation (1 row per regime×variant) ----------
    def _agg_run(g: pd.DataFrame):
        # shared/private means from GT-CCA
        shared = g[g["latent_type"].isin(["Shared1", "Shared2"])]
        priv   = g[g["latent_type"].isin(["Private1", "Private2"])]

        out = {
            "mean_cca_shared":  shared["mean_cca"].mean(),
            "mean_cca_private": priv["mean_cca"].mean(),
            "mean_angle_shared":  shared["mean_angle"].mean(),
            "mean_angle_private": priv["mean_angle"].mean(),
            "mean_mse_shared":  shared["mean_mse"].mean(),
            "mean_mse_private": priv["mean_mse"].mean(),
        }

        # values repeated per row → just take the first non-null
        def first(col):
            s = g[col].dropna()
            return s.iloc[0] if len(s) else np.nan

        # alignment & cross-pred R²
        for c in [
            "cca_shared12",
            "r2_x_full","r2_y_full",
            "r2_x_shared_only","r2_y_shared_only",
            "r2_x_private_only","r2_y_private_only",
            "r2_y_from_x_shared","r2_x_from_y_shared",
        ]:
            if c in g.columns:
                out[c] = first(c)

        # leakage (avg across regions if present)
        leak_terms = {
            "leak_shared_vs_gt_private":
                np.nanmean([g.get("cca_leak_s1_vs_gtp1", pd.Series(dtype=float)).mean(),
                            g.get("cca_leak_s2_vs_gtp2", pd.Series(dtype=float)).mean()]),
            "leak_private_vs_gt_shared":
                np.nanmean([g.get("cca_leak_p1_vs_gts1", pd.Series(dtype=float)).mean(),
                            g.get("cca_leak_p2_vs_gts2", pd.Series(dtype=float)).mean()]),
        }
        out.update(leak_terms)

        # optional composite score (simple, transparent):
        # emphasize shared CCA + alignment + cross-R², penalize leakage
        shared_cross_r2 = np.nanmean([out.get("r2_y_from_x_shared", np.nan),
                                      out.get("r2_x_from_y_shared", np.nan)])
        leak_pen = np.nanmean([out.get("leak_shared_vs_gt_private", np.nan),
                               out.get("leak_private_vs_gt_shared", np.nan)])
        out["composite_score"] = (
            0.45 * out["mean_cca_shared"]
          + 0.25 * out.get("cca_shared12", np.nan)
          + 0.20 * shared_cross_r2
          - 0.10 * (leak_pen if not np.isnan(leak_pen) else 0.0)
        )
        return pd.Series(out)

    run_agg = (summary_df
               .groupby(["regime","variant","seed_label"], as_index=False)
               .apply(_agg_run)
               .reset_index(drop=True))
    
    # Add derived columns you defined
    run_agg = _add_derived_columns(run_agg)

    # ---------- write outputs ----------
    os.makedirs(os.path.dirname(save_excel_path), exist_ok=True)

    # Try xlsxwriter → openpyxl → CSV fallback
    writer = None
    try:
        writer = pd.ExcelWriter(save_excel_path, engine="xlsxwriter")
    except Exception:
        try:
            writer = pd.ExcelWriter(save_excel_path, engine="openpyxl")
            print("[INFO] Using openpyxl engine for Excel output.")
        except Exception as e:
            print(f"[WARN] Excel engines unavailable ({e}). Writing CSVs instead.")
            base = os.path.splitext(save_excel_path)[0]
            summary_df.to_csv(base + "_summary.csv", index=False)
            run_agg.to_csv(base + "_run_agg.csv", index=False)
            print(f"[DONE] Wrote CSV summaries with base path: {base}")
            return summary_df

    with writer:
        summary_df.to_excel(writer, index=False, sheet_name="summary_all")
        run_agg.to_excel(writer, index=False, sheet_name="run_agg")

        # Optional: per-latent sheets
        for lt in ["Shared1", "Shared2", "Private1", "Private2"]:
            sub = summary_df[summary_df["latent_type"] == lt]
            if len(sub) > 0:
                sub.to_excel(writer, index=False, sheet_name=lt)

    print(f"[DONE] Wrote evaluation summary to: {save_excel_path}")
    return summary_df