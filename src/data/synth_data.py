from dataclasses import dataclass,asdict
from typing import Dict, Tuple, List, Optional, Any
import numpy as np
import os

@dataclass
class LeadGeometry:
    # e.g., STN example: [3,3,2,4] rows of contacts
    contacts_per_row: List[int]
    # optional explicit bipolar pairs (1-indexed contacts). If None, auto-generate neighbor bipolars.
    bipolar_pairs: Optional[List[Tuple[int,int]]] = None

def _rng(seed):
    return np.random.default_rng(seed) if seed is not None else np.random.default_rng()

def _make_contact_indices(contacts_per_row: List[int]) -> Dict[int, Tuple[int,int]]:
    """
    Return mapping: contact_id (1-indexed) -> (row_idx, within_row_idx)
    """
    mapping = {}
    cid = 1
    for r, n in enumerate(contacts_per_row):
        for j in range(n):
            mapping[cid] = (r, j)
            cid += 1
    return mapping

def _auto_bipolar_pairs(contacts_per_row: List[int]) -> List[Tuple[int,int]]:
    """
    Simple neighbor bipolars within each row (1-2, 2-3, ...). Cross-row pairs are not made automatically.
    """
    pairs = []
    base = 1
    for n in contacts_per_row:
        for j in range(n-1):
            pairs.append((base+j, base+j+1))
        base += n
    return pairs

def _bipolarize(contacts_ts: np.ndarray, pairs: List[Tuple[int,int]]) -> np.ndarray:
    """
    contacts_ts: (T, C_contacts)
    return: (T, C_bipolar)
    """
    T, Cc = contacts_ts.shape
    B = np.zeros((len(pairs), Cc))
    for i,(a,b) in enumerate(pairs):
        B[i, a-1] = 1.0
        B[i, b-1] = -1.0
    return contacts_ts @ B.T, B  # (T, Cb), (Cb, Cc)

def _one_over_f(T, rng, strength=0.3):
    """
    crude 1/f-ish noise by filtering white noise in freq domain
    """
    w = rng.standard_normal(T)
    W = np.fft.rfft(w)
    freqs = np.fft.rfftfreq(T, d=1.0)
    denom = np.maximum(freqs, 1.0/T) ** 0.5  # ~1/sqrt(f)
    Wf = W / denom
    y = np.fft.irfft(Wf, n=T)
    y = y / (np.std(y)+1e-8)
    return strength * y

def _apply_ar1(x, rho):
    """
    x: (T, C)
    """
    y = np.zeros_like(x)
    y[0] = x[0]
    for t in range(1, x.shape[0]):
        y[t] = rho*y[t-1] + np.sqrt(1-rho**2)*x[t]
    return y

def _nonlinear_mix(s, p, mode, g, beta):
    if mode == "static":
        x = np.tanh(0.5*(s+p)) + 0.1*(s+p)
    elif mode == "gain":
        x = s + (1.0 + g*np.tanh(s)) * p
        x = x + 0.05*np.tanh(s)
    elif mode == "bilinear":
        cross = beta * (s*p)
        x = np.tanh(0.4*(s + p + cross)) + 0.1*(s + p)
    elif mode == "gain+bilinear":
        x = s + (1.0 + g*np.tanh(s)) * p
        x = x + beta * (s*p)
        x = np.tanh(0.3*x) + 0.1*x
    else:
        raise ValueError(f"Unknown mode {mode}")
    return x

def _col_normalize(W, eps=1e-8):
    nrm = np.linalg.norm(W, axis=0, keepdims=True)
    return W / np.maximum(nrm, eps)

def _row_common(time_T, geom, strength, rng):
    if strength <= 1e-6:
        return np.zeros((time_T, sum(geom.contacts_per_row)))
    mapping = _make_contact_indices(geom.contacts_per_row)
    rows = len(geom.contacts_per_row)
    row_ts = []
    for _ in range(rows):
        drift = _one_over_f(time_T, rng, strength=1.0)
        drift = 0.5 * _apply_ar1(drift[:, None], rho=0.9).squeeze()
        row_ts.append(drift)
    row_ts = np.stack(row_ts, axis=1)  # (T, rows)
    out = np.zeros((time_T, sum(geom.contacts_per_row)))
    for cid in range(1, sum(geom.contacts_per_row) + 1):
        r_idx, _ = mapping[cid]
        out[:, cid - 1] = strength * row_ts[:, r_idx]
    return out

def _make_spatial_mixing_gaussian(C_contacts: int, contacts_per_row: List[int],
                                  rng, spatial_decay=1.5, cross_row_leak=0.1):
    # smooth kernel on a grid
    mapping = _make_contact_indices(contacts_per_row)
    coords = np.zeros((C_contacts, 2))
    for cid, (r, j) in mapping.items():
        coords[cid-1] = np.array([j, r])
    D = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    K = np.exp(-(D**2) / (2 * spatial_decay**2))
    # light cross-row leakage
    same_row = np.zeros_like(D)
    for i in range(C_contacts):
        for j in range(C_contacts):
            same_row[i, j] = 1.0 if mapping[i+1][0] == mapping[j+1][0] else cross_row_leak
    K = K * same_row + 1e-6 * np.eye(C_contacts)
    K = K / (K.sum(axis=1, keepdims=True) + 1e-12)
    return K

def _bursty_latent(T, fs, rng, freq_base=10., burst_prob=0.15, burst_amp=0.6):
    t = np.arange(T) / fs
    phase = rng.uniform(0, 2*np.pi)
    freq = freq_base + rng.uniform(-2, 2)
    x = np.sin(2*np.pi*freq*t + phase)
    if rng.random() < burst_prob:
        L = rng.integers(10, 50)
        s = rng.integers(0, T-L)
        # 30–40 Hz burst
        fb = 30 + 10*rng.random()
        x[s:s+L] += burst_amp*np.sin(2*np.pi*fb*t[:L] + phase)
    # occasional freq jump
    if rng.random() < 0.5:
        s2 = rng.integers(T//3, 2*T//3)
        f2 = freq_base + rng.uniform(8, 18)
        x[s2:] = np.sin(2*np.pi*f2*t[:T-s2] + phase)
    return x

# ---- Add these helpers ----
def _zscore(a, axis=-1, eps=1e-8):
    m = a.mean(axis=axis, keepdims=True)
    s = a.std(axis=axis, keepdims=True) + eps
    return (a - m) / s

def _warp_shared_latents(shared, region=1, mode="none"):
    """
    shared: (Ds, T). Apply region-specific monotone-but-different warps.
    """
    if mode == "none":
        return shared
    z = _zscore(shared, axis=1)
    if mode == "region_mismatch":
        if region == 1:
            y = np.tanh(0.9*z) + 0.15*z + 0.10*(z**2)   # monotone-ish, mild quadratic
        else:
            y = np.tanh(0.5*z) + 0.20*z - 0.10*(z**3)   # different shape
    elif mode == "cubic":
        y = z + 0.5*(z**3)
    else:
        raise ValueError("unknown warp mode")
    # renormalize to preserve per-dim variance
    y = y * (shared.std(axis=1, keepdims=True) / (y.std(axis=1, keepdims=True)+1e-8))
    return y

def _time_varying_delay(arr, k_t):
    """arr: (D,T), k_t: (T,) ints; roll each timepoint by its own lag."""
    D, T = arr.shape
    out = np.zeros_like(arr)
    for t in range(T):
        out[:, t] = arr[:, (t - int(k_t[t])) % T]
    return out


def generate_synth_data(
    # core sizes
    n_trials=100, T=250, fs=500, shared_dim=3, private_dim=3,
    # geometry & montage
    geom_region1=None, geom_region2=None,
    bipolarize=True,
    # mixing (choose one)
    spatial_mixing: str = "none",     # {"none","gaussian","random"}
    spatial_decay=1.5, cross_row_leak=0.1,
    normalize_loadings=True, mix_scale_shared=1.0, mix_scale_private=1.0,
    # latents & nonlinearity
    nonlin_mode="static", gain_g=0.25, bilinear_beta=0.08,
    interaction_strength=0.0,         # pre-nonlinearity s*p (usually 0)
    # stationarity (no per-trial rotations/delays)
    shared_variant: str = "fixed_gain", # {"identity","fixed_gain"}
    fixed_gain_range=(0.9, 1.1),      # only used if shared_variant="fixed_gain"
    # noises (independent toggles)
    add_one_over_f=False, one_over_f_strength=0.3,
    ar_stage="none",                   # {"none","latents","channels"}
    ar1_rho=0.6,
    sensor_noise=0.02,
    add_row_common=False, row_common_strength=0.0,
    add_common_mode=False, common_mode_rank=1, common_mode_strength=0.3,
    # reproducibility
    seed: Optional[int] = 123,
    shared_warp="none",           # {"none","region_mismatch","cubic"}
    heteroscedastic_noise=0.0,    # 0 = off, else multiple options: scales noise ~ (1 + h*|s_norm|), power, multiplixative or combined
    het_driver = "shared",   # {"shared","pre","post"}
    het_mode   = "power",    # {"abs","power","mult","power+mult"}
    timevary_delay=False,         # False/True (applies to shared2)
    tvd_amplitude=3,              # max integer lag (set LAGS >= tvd_amplitude+1)
    tvd_cycles=1.0,               # cycles per trial of the lag profile
) -> Dict[str, np.ndarray]:
    """
    Stationary generator with additive options.
    Outputs match SPIRE pipeline schema.
    First we generate the weights for shared and private latents. Then for each trial we do the following:
    we generate shared1 and shared2 latents.
    optionally Apply region-specific monotone-but-different warps.
    optional time-varying delay on shared2
    we genrate private1 and private2 latents.
    optionally we add 1/f to private latents.
    optionally we add AR(1) noise to both latent types.
    by multiplying weights to latents we make their parts for observations (s1,s2,p1,p2)
    optionally we include interation term between s1, p1 and s2,p2
    optionally we mix them nonlineraly to make the observation (x1, x2)
    either apply homoscedastic or heteroscedastic noise driven by shared amplitude
    optionally we add common noise
    optionally we add channel level AR(1) noise
    optionally we bipolarize the channels.
    """
    rng = _rng(seed)

    # --- geometry
    if geom_region1 is None:
        geom_region1 = LeadGeometry([3,3,2,2], bipolar_pairs=[(1,2),(1,3),(2,3),(4,5),(4,6),(5,6),(7,8),(9,10)])
    if geom_region2 is None:
        geom_region2 = LeadGeometry([3,3,2,2], bipolar_pairs=[(1,2),(1,3),(2,3),(4,5),(4,6),(5,6),(7,8),(9,10)])

    def setup_geom(geom: 'LeadGeometry'):
        C_contacts = sum(geom.contacts_per_row)
        pairs = geom.bipolar_pairs if geom.bipolar_pairs is not None else _auto_bipolar_pairs(geom.contacts_per_row)
        return C_contacts, pairs
    C1_contacts, pairs1 = setup_geom(geom_region1)
    C2_contacts, pairs2 = setup_geom(geom_region2)

    # --- mixing matrices W_* (contacts × dims)
    if spatial_mixing == "gaussian":
        K1 = _make_spatial_mixing_gaussian(C1_contacts, geom_region1.contacts_per_row, rng,
                                           spatial_decay=spatial_decay, cross_row_leak=cross_row_leak)
        K2 = _make_spatial_mixing_gaussian(C2_contacts, geom_region2.contacts_per_row, rng,
                                           spatial_decay=spatial_decay, cross_row_leak=cross_row_leak)
        W_shared1 = K1 @ rng.standard_normal((C1_contacts, shared_dim))
        W_private1 = K1 @ rng.standard_normal((C1_contacts, private_dim))
        W_shared2 = K2 @ rng.standard_normal((C2_contacts, shared_dim))
        W_private2 = K2 @ rng.standard_normal((C2_contacts, private_dim))
    elif spatial_mixing == "random":
        W_shared1 = rng.standard_normal((C1_contacts, shared_dim))
        W_private1 = rng.standard_normal((C1_contacts, private_dim))
        W_shared2 = rng.standard_normal((C2_contacts, shared_dim))
        W_private2 = rng.standard_normal((C2_contacts, private_dim))
    elif spatial_mixing == "none":
        # per-contact sparse-ish identity-style mixing; simple & SNR-friendly
        def simple_W(C, D):
            W = rng.standard_normal((C, D))
            return W
        W_shared1 = simple_W(C1_contacts, shared_dim)
        W_private1 = simple_W(C1_contacts, private_dim)
        W_shared2 = simple_W(C2_contacts, shared_dim)
        W_private2 = simple_W(C2_contacts, private_dim)
    else:
        raise ValueError("spatial_mixing must be one of {'none','gaussian','random'}")

    if normalize_loadings:
        W_shared1 = _col_normalize(W_shared1) * mix_scale_shared
        W_shared2 = _col_normalize(W_shared2) * mix_scale_shared
        W_private1 = _col_normalize(W_private1) * mix_scale_private
        W_private2 = _col_normalize(W_private2) * mix_scale_private

    # --- containers
    region1_data, region2_data = [], []
    contact1_all, contact2_all = [], []
    gt_shared_base, gt_shared1_all, gt_shared2_all = [], [], []
    gt_private1_all, gt_private2_all = [], []
    per_trial_mods = []

    # --- trial loop (stationary: no per-trial R/d)
    for tr in range(n_trials):
        # shared base latents
        base = np.stack([_bursty_latent(T, fs, rng) for _ in range(shared_dim)], axis=0)
        if add_one_over_f:
            base += np.stack([_one_over_f(T, rng, strength=one_over_f_strength) for _ in range(shared_dim)], axis=0)

        if shared_variant == "identity":
            shared1 = base.copy()
            shared2 = base.copy()
            mods = {"g1": np.ones(shared_dim), "g2": np.ones(shared_dim)}
        elif shared_variant == "fixed_gain":
            g1 = rng.uniform(*fixed_gain_range, size=shared_dim)
            g2 = rng.uniform(*fixed_gain_range, size=shared_dim)
            shared1 = g1[:, None] * base
            shared2 = g2[:, None] * base
            mods = {"g1": g1, "g2": g2}
        else:
            raise ValueError("shared_variant must be {'identity','fixed_gain'}")
        
        # NEW: region-mismatched warp of shared (breaks linear cross-region mapping)
        shared1 = _warp_shared_latents(shared1, region=1, mode=shared_warp)
        shared2 = _warp_shared_latents(shared2, region=2, mode=shared_warp)

        # NEW: optional time-varying delay on shared2 (same profile every trial)
        if timevary_delay and tvd_amplitude > 0:
            t = np.arange(T)
            k_t = (tvd_amplitude * np.sin(2*np.pi*tvd_cycles * t / T)).round()
            shared2 = _time_varying_delay(shared2, k_t)

        # private latents
        priv1 = np.stack([_bursty_latent(T, fs, rng, freq_base=8+3*i) for i in range(private_dim)], axis=0)
        priv2 = np.stack([_bursty_latent(T, fs, rng, freq_base=9+3*i) for i in range(private_dim)], axis=0)
        if add_one_over_f:
            priv1 += np.stack([_one_over_f(T, rng, strength=one_over_f_strength) for _ in range(private_dim)], axis=0)
            priv2 += np.stack([_one_over_f(T, rng, strength=one_over_f_strength) for _ in range(private_dim)], axis=0)

        if ar_stage == "latents" and ar1_rho > 0:
            def ar_lat(Z): return _apply_ar1(Z.T, rho=ar1_rho).T
            shared1, shared2 = ar_lat(shared1), ar_lat(shared2)
            priv1,   priv2   = ar_lat(priv1),   ar_lat(priv2)

        # mix to contacts
        s1 = (W_shared1 @ shared1).T; p1 = (W_private1 @ priv1).T
        s2 = (W_shared2 @ shared2).T; p2 = (W_private2 @ priv2).T

        # optional pre-nonlinearity cross-term
        if interaction_strength > 0:
            s1 = s1 + interaction_strength * (s1 * p1)
            s2 = s2 + interaction_strength * (s2 * p2)

        # nonlinearity
        x1 = _nonlinear_mix(s1, p1, nonlin_mode, gain_g, bilinear_beta)
        x2 = _nonlinear_mix(s2, p2, nonlin_mode, gain_g, bilinear_beta)

        # ---- HETEROSCEDASTIC / MULTIPLICATIVE NOISE ----
        # Choose what drives the heteroscedasticity:
        min_sigma  = 1e-6       # floor on σ
        max_mult   = 6.0        # cap σ ≤ max_mult * sensor_noise

        if heteroscedastic_noise > 0.0:
            # pick the driver signal per region (shape T×C)
            if het_driver == "shared":
                driver1, driver2 = s1, s2             # shared component before NL
            elif het_driver == "pre":
                driver1, driver2 = (s1 + p1), (s2 + p2)  # before NL
            elif het_driver == "post":
                driver1, driver2 = x1, x2             # after NL (pre-noise)
            else:
                raise ValueError("het_driver must be {'shared','pre','post'}")

            z1 = _zscore(driver1, axis=0); z2 = _zscore(driver2, axis=0)

            # default σ fields (T×C)
            if het_mode == "abs":
                sigma1 = sensor_noise * (1.0 + heteroscedastic_noise * np.abs(z1))
                sigma2 = sensor_noise * (1.0 + heteroscedastic_noise * np.abs(z2))

            elif het_mode == "power":
                # variance grows with power; much stronger than |z|
                sigma1 = sensor_noise * np.sqrt(1.0 + heteroscedastic_noise * (z1**2))
                sigma2 = sensor_noise * np.sqrt(1.0 + heteroscedastic_noise * (z2**2))

            elif het_mode == "mult":
                # pure multiplicative jitter on the signal (plus baseline σ0)
                x1 *= (1.0 + heteroscedastic_noise * rng.standard_normal(x1.shape))
                x2 *= (1.0 + heteroscedastic_noise * rng.standard_normal(x2.shape))
                sigma1 = np.full_like(x1, sensor_noise)
                sigma2 = np.full_like(x2, sensor_noise)

            elif het_mode == "power+mult":
                sigma1 = sensor_noise * np.sqrt(1.0 + heteroscedastic_noise * (z1**2))
                sigma2 = sensor_noise * np.sqrt(1.0 + heteroscedastic_noise * (z2**2))
                # a small extra multiplicative part
                x1 *= (1.0 + 0.5 * heteroscedastic_noise * rng.standard_normal(x1.shape))
                x2 *= (1.0 + 0.5 * heteroscedastic_noise * rng.standard_normal(x2.shape))
            else:
                raise ValueError("het_mode must be one of {'abs','power','mult','power+mult'}")

            # clamp σ to avoid numerical blow-ups
            sigma1 = np.clip(sigma1, min_sigma, max_mult * max(1e-12, sensor_noise))
            sigma2 = np.clip(sigma2, min_sigma, max_mult * max(1e-12, sensor_noise))

            # add the heteroscedastic noise
            x1 += rng.standard_normal(x1.shape) * sigma1
            x2 += rng.standard_normal(x2.shape) * sigma2

        else:
            if sensor_noise > 0:
                x1 += sensor_noise * rng.standard_normal(x1.shape)
                x2 += sensor_noise * rng.standard_normal(x2.shape)

        # row-common slow drift (pre-bipolar)
        if add_row_common and row_common_strength > 0:
            x1 += _row_common(T, geom_region1, row_common_strength, rng)
            x2 += _row_common(T, geom_region2, row_common_strength, rng)

        # low-rank common mode (pre-bipolar)
        if add_common_mode and common_mode_strength > 0:
            def common_mode(time_T, C_contacts, rank):
                U = rng.standard_normal((C_contacts, rank))
                U, _ = np.linalg.qr(U)
                z = np.stack([_one_over_f(time_T, rng, strength=1.0) for _ in range(rank)], axis=1)
                return z @ U.T
            x1 += common_mode_strength * common_mode(T, C1_contacts, common_mode_rank)
            x2 += common_mode_strength * common_mode(T, C2_contacts, common_mode_rank)

        if ar_stage == "channels" and ar1_rho > 0:
            x1 = _apply_ar1(x1, rho=ar1_rho)
            x2 = _apply_ar1(x2, rho=ar1_rho)

        # # sensor noise
        # if sensor_noise > 0:
        #     x1 += sensor_noise * rng.standard_normal(x1.shape)
        #     x2 += sensor_noise * rng.standard_normal(x2.shape)

        # bipolarization
        if bipolarize:
            xb1, B1 = _bipolarize(x1, pairs1)
            xb2, B2 = _bipolarize(x2, pairs2)
        else:
            xb1, xb2 = x1, x2
            B1 = np.eye(x1.shape[1]); B2 = np.eye(x2.shape[1])

        # collect
        region1_data.append(xb1);   region2_data.append(xb2)
        contact1_all.append(x1);    contact2_all.append(x2)
        gt_shared_base.append(base)
        gt_shared1_all.append(shared1); gt_shared2_all.append(shared2)
        gt_private1_all.append(priv1);  gt_private2_all.append(priv2)
        per_trial_mods.append(mods)  # simple, no per-trial rotation/delay

    out = {
        "region1": np.stack(region1_data, axis=0),
        "region2": np.stack(region2_data, axis=0),
        "gt_shared_base": np.stack(gt_shared_base, axis=0),
        "gt_shared1": np.stack(gt_shared1_all, axis=0),
        "gt_shared2": np.stack(gt_shared2_all, axis=0),
        "gt_private1": np.stack(gt_private1_all, axis=0),
        "gt_private2": np.stack(gt_private2_all, axis=0),
        "W_shared1": W_shared1, "W_private1": W_private1,
        "W_shared2": W_shared2, "W_private2": W_private2,
        "pairs_region1": pairs1, "pairs_region2": pairs2,
        "geom_region1": asdict(geom_region1),
        "geom_region2": asdict(geom_region2),
        "per_trial_mods": per_trial_mods,
        "config": {
            "n_trials": n_trials, "T": T, "fs": fs,
            "shared_dim": shared_dim, "private_dim": private_dim,
            "bipolarize": bipolarize, "spatial_mixing": spatial_mixing,
            "nonlin_mode": nonlin_mode, "gain_g": gain_g, "bilinear_beta": bilinear_beta,
            "add_one_over_f": add_one_over_f, "one_over_f_strength": one_over_f_strength,
            "ar_stage": ar_stage, "ar1_rho": ar1_rho, "sensor_noise": sensor_noise,
            "add_row_common": add_row_common, "row_common_strength": row_common_strength,
            "add_common_mode": add_common_mode, "common_mode_rank": common_mode_rank,
            "common_mode_strength": common_mode_strength, "seed": seed
        }
    }
    # expose contacts if you want:
    out["contacts_region1"] = np.stack(contact1_all, axis=0)
    out["contacts_region2"] = np.stack(contact2_all, axis=0)
    return out


def load_saved_dataset(regime_name, config, data_dir):
    npz_path = os.path.join(data_dir, f"{regime_name}.npz")

    if os.path.exists(npz_path):
        data = dict(np.load(npz_path, allow_pickle=True))
    else:
        raise FileNotFoundError(f"No saved file found for {regime_name}")

    return {"regime": regime_name, "config": config, "data": data}