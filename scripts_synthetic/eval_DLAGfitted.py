#the DLAG models are fitted to synthetic data with 4 seeds in MATLAB and here we evaluate them similar to what we did with SPIRE, so then the excels are used in R to do statistical analysis
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]   # repo root
sys.path.insert(0, str(root / "src"))

import os
import numpy as np
import scipy
import torch
import pandas as pd

from evaluate_synth import compare_latent_performance, mean_cca

#reload from npz (preferred) or .mat
def load_saved_dataset(regime_name, config, data_dir):
    npz_path = os.path.join(data_dir, f"{regime_name}.npz")

    if os.path.exists(npz_path):
        data = dict(np.load(npz_path, allow_pickle=True))
    else:
        raise FileNotFoundError(f"No saved file found for {regime_name}")

    return {"regime": regime_name, "config": config, "data": data}

def build_dataset(regime_name, cfg, data_save_dir):
    ds = load_saved_dataset(regime_name, cfg, data_save_dir)  # <- your helper
    # enforce structure: {"regime": name, "data": ...}
    assert "regime" in ds and "data" in ds, "load_saved_dataset must return {'regime','data'}"
    return ds

# --- helper to load one DLAG latent struct from .mat ---
def load_dlag_latents_struct(mat_path):
    mat = scipy.io.loadmat(mat_path)
    if "dlag_latents_struct" not in mat:
        raise KeyError(f"'dlag_latents_struct' not found in {mat_path}")
    return mat["dlag_latents_struct"]  # pass through; evaluation handles shapes

@torch.no_grad()
def evaluate_one_model_on_one_dataset_DLAG(latents, data, run_name):
    """
    1) load latents
    4) metrics vs GT (CCA/angles/MSE) + extra diagnostics:
       - Reconstruction R² per region (full / shared-only / private-only)
       - Cross-prediction R² (Y from shared-X only; X from shared-Y only)
       - Leakage CCAs (shared vs GT-private; private vs GT-shared)
       - Shared alignment (CCA between learned shared1 and shifted shared2)
    """
    shared_r1  = latents['g1_shared'][0,0]    # shape: (nTrials, 4, T)
    private_r1 = latents['g1_private'][0,0]   # shape: (nTrials, 2, T)
    shared_r2  = latents['g2_shared'][0,0]    # shape: (nTrials, 4, T)
    private_r2 = latents['g2_private'][0,0]   # shape: (nTrials, 2, T)

    # Permute axes to (nTrials, T, dim)
    shared_r1  = np.transpose(shared_r1, (0, 2, 1))
    private_r1 = np.transpose(private_r1, (0, 2, 1))
    shared_r2  = np.transpose(shared_r2, (0, 2, 1))
    private_r2 = np.transpose(private_r2, (0, 2, 1))

    # -------- GT comparisons (existing) --------
    gt_s1 = data["gt_shared1"]; gt_s2 = data["gt_shared2"]
    gt_p1 = data["gt_private1"]; gt_p2 = data["gt_private2"]

    _, m_shared1 = compare_latent_performance(shared_r1, gt_s1, model_name=f"{run_name}_Shared1")
    _, m_shared2 = compare_latent_performance(shared_r2, gt_s2, model_name=f"{run_name}_Shared2")
    _, m_priv1   = compare_latent_performance(private_r1, gt_p1, model_name=f"{run_name}_Private1")
    _, m_priv2   = compare_latent_performance(private_r2, gt_p2, model_name=f"{run_name}_Private2")

    # -------- extra: leakage checks (shared↔private GT) --------
    Tm = shared_r1.shape[1]  # T_model
    gt_p1_T = np.transpose(gt_p1, (0,2,1))[:, -Tm:, :]
    gt_s1_T = np.transpose(gt_s1, (0,2,1))[:, -Tm:, :]
    gt_p2_T = np.transpose(gt_p2, (0,2,1))[:, -Tm:, :]
    gt_s2_T = np.transpose(gt_s2, (0,2,1))[:, -Tm:, :]

    mean_cca_s1_vs_gtp1, _ = mean_cca(shared_r1, gt_p1_T, n_comp=shared_r1.shape[-1])
    mean_cca_p1_vs_gts1, _ = mean_cca(private_r1, gt_s1_T, n_comp=private_r1.shape[-1])
    mean_cca_s2_vs_gtp2, _ = mean_cca(shared_r2, gt_p2_T, n_comp=shared_r2.shape[-1])
    mean_cca_p2_vs_gts2, _ = mean_cca(private_r2, gt_s2_T, n_comp=private_r2.shape[-1])

    # -------- extra: shared alignment across regions --------
    mean_cca_shared12, _ = mean_cca(shared_r1, shared_r2, n_comp=shared_r2.shape[-1])

    # -------- assemble tidy rows (repeat run-level extras on each latent row) --------
    common = {
        # leakage CCAs
        "cca_leak_s1_vs_gtp1": mean_cca_s1_vs_gtp1,
        "cca_leak_p1_vs_gts1": mean_cca_p1_vs_gts1,
        "cca_leak_s2_vs_gtp2": mean_cca_s2_vs_gtp2,
        "cca_leak_p2_vs_gts2": mean_cca_p2_vs_gts2,

        # shared alignment
        "cca_shared12": mean_cca_shared12,
    }

    rows = [
        {"latent_type": "Shared1", **m_shared1, **common},
        {"latent_type": "Shared2", **m_shared2, **common},
        {"latent_type": "Private1", **m_priv1,   **common},
        {"latent_type": "Private2", **m_priv2,   **common},
    ]
    return rows

def add_derived_columns_DLAG(run_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Extends run_agg with:
      - slgs_signal_geom_mean   (new; geometric mean of signal block)
      - slgs_antileak_geom_mean (new; geometric mean of anti-leak block)
      - score_slgs              (new; final balanced score in [0,1])

    SLGS (balanced, disentanglement-focused):
        score_slgs =
            (s1^(1/3) * s12^(1/3) * sp^(1/3)) *
            ((1-lsp)^(1/2) * (1-lps)^(1/2))

    Where columns (if missing) are safely treated as NaN:
        s1   = mean_cca_shared                  in [0,1]
        s12  = cca_shared12                     in [0,1]
        sp   = mean_cca_private                 in [0,1]
        lsp  = leak_shared_vs_gt_private        in [0,1]
        lps  = leak_private_vs_gt_shared        in [0,1]
    """

    df = run_agg.copy()

    # Ensure required columns exist (fill with NaN if missing)
    for c in [
        "mean_cca_shared", "cca_shared12", "mean_cca_private",
        "leak_shared_vs_gt_private", "leak_private_vs_gt_shared"
    ]:
        if c not in df.columns:
            df[c] = np.nan

    # ---- New: Balanced SLGS score for disentanglement ----
    eps = 1e-6
    s1  = np.clip(df["mean_cca_shared"].astype(float).to_numpy(), 0.0, 1.0)
    s12 = np.clip(df["cca_shared12"].astype(float).to_numpy(),     0.0, 1.0)
    sp  = np.clip(df["mean_cca_private"].astype(float).to_numpy(), 0.0, 1.0)
    lsp = np.clip(df["leak_shared_vs_gt_private"].astype(float).to_numpy(), 0.0, 1.0)
    lps = np.clip(df["leak_private_vs_gt_shared"].astype(float).to_numpy(), 0.0, 1.0)

    # Avoid exact zeros in geometric products
    s1  = np.clip(s1,  eps, 1.0 - eps)
    s12 = np.clip(s12, eps, 1.0 - eps)
    sp  = np.clip(sp,  eps, 1.0 - eps)
    one_minus_lsp = np.clip(1.0 - lsp, eps, 1.0)
    one_minus_lps = np.clip(1.0 - lps, eps, 1.0)

    # Balanced exponents (shared, alignment, private equally important; symmetric leak penalties)
    alpha = beta = gamma = 1/3
    delta = epsilon = 1/2

    signal_geom   = (s1 ** alpha) * (s12 ** beta) * (sp ** gamma)
    antileak_geom = (one_minus_lsp ** delta) * (one_minus_lps ** epsilon)
    score_slgs    = signal_geom * antileak_geom

    df["slgs_signal_geom_mean"]   = signal_geom
    df["slgs_antileak_geom_mean"] = antileak_geom
    df["score_slgs"]              = score_slgs

    return df

def evaluate_oneModel_all_datasets_DLAG(
    latents,
    ALL_DATASETS,
    save_excel_path
):
    """
    Loops over all trained runs and matching datasets (by regime),
    extracts & saves latents, evaluates metrics, and writes a multi-sheet Excel.
    """
    all_rows = []


    for item in ALL_DATASETS:
        regime  = item["regime"]
        data = item["data"]
        run_name = f"{regime}_DLAG"

        try:
            rows = evaluate_one_model_on_one_dataset_DLAG(
                latents=latents,
                data=data,
                run_name=run_name,
            )
            #DLAG here checked
            for r in rows:
                r.update({"regime": regime, "variant": "DLAG"})
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
    meta_cols = ["regime", "variant", "latent_type"]
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
            "cca_shared12"
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
        leak_pen = np.nanmean([out.get("leak_shared_vs_gt_private", np.nan),
                               out.get("leak_private_vs_gt_shared", np.nan)])
        out["composite_score"] = (
            0.45 * out["mean_cca_shared"]
          + 0.25 * out.get("cca_shared12", np.nan)
          - 0.10 * (leak_pen if not np.isnan(leak_pen) else 0.0)
        )
        return pd.Series(out)

    run_agg = (summary_df
               .groupby(["regime","variant"], as_index=False)
               .apply(_agg_run)
               .reset_index(drop=True))
    
    # Add derived columns you defined
    run_agg = add_derived_columns_DLAG(run_agg)

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



#--------- Main--------
DLAG_fitted_path = r"F:\comp_project\synthecticData\DLAG_fitted"
data_save_dir = r"F:\comp_project\synthecticData\dataT"
eval_out_dir      = r"F:\comp_project\synthecticData\evaluation"

DATA_SETS = {
    #D0 linear mixing with Gaussian noise
    "D0_linear": dict(
        n_trials=100, T=250, fs=500, shared_dim=3, private_dim=3,
        bipolarize=True,
        spatial_mixing="none",
        nonlin_mode="static",
        shared_variant="identity",
        add_one_over_f=False, ar_stage="none",
        sensor_noise=0.01,
        add_row_common=False, add_common_mode=False,
        seed=701,
    ),
    
    #D1 adds region-mismatched nonlinear warps and bilinear mixing with 1/f noise and AR(1) latents
    "D1_warp_nonLinear": dict(
        n_trials=100, T=250, fs=500, shared_dim=3, private_dim=3,
        bipolarize=True,
        spatial_mixing="random",              # or "gaussian" if you want harsher bipolar effects
        nonlin_mode="gain+bilinear", gain_g=0.8, bilinear_beta=0.8,
        interaction_strength=0.2,
        shared_variant="identity",
        shared_warp="region_mismatch",
        add_one_over_f=True, one_over_f_strength=0.2,
        ar_stage="latents", ar1_rho=0.3,
        sensor_noise=0.02,
        add_row_common=False, add_common_mode=False,
        seed=702,
    ),
    
    # D2 extends D1 with a slow time-varying inter-regional lag (sinusoidal) to model phase-dependent communication and latency variability
    "D2_timevary_delay": dict(
        n_trials=100, T=250, fs=500, shared_dim=3, private_dim=3,
        bipolarize=True,
        spatial_mixing="random",              # or "gaussian" if you want harsher bipolar effects
        nonlin_mode="gain+bilinear", gain_g=0.8, bilinear_beta=0.8,
        interaction_strength=0.2,
        shared_variant="identity",
        shared_warp="region_mismatch",
        timevary_delay=True, tvd_amplitude=3, tvd_cycles=1.0,   # <-- NEW
        add_one_over_f=True, one_over_f_strength=0.2,
        ar_stage="latents", ar1_rho=0.3,
        sensor_noise=0.02,
        add_row_common=False, add_common_mode=False,
        seed=702,
    ),
}

#when we wanted to fit the data with some seeds the model would give error so we used diferent seeds!
DLAG_FILES = { 
    "D0_linear": [
        "D0_linear_easy_500Iter_jitter_seed705.mat",
        "D0_linear_easy_500Iter_jitter_seed701.mat",
        "D0_linear_easy_500Iter_jitter.mat",
        "D0_linear_easy_500Iter_jitter_seed45"
    ],
    "D1_warp_nonLinear": [
        "D1_warp_gainbilin_500Iter_jitter_seed703.mat",
        "D1_warp_gainbilin_500Iter_jitter_seed701.mat",
        "D1_warp_gainbilin_500Iter_jitter.mat",
        "D1_warp_gainbilin_500Iter_jitter_seed45"
    ],
    "D2_timevary_delay": [
        "D2_timevary_delay_500Iter_jitter_seed702.mat",
        "D2_timevary_delay_500Iter_jitter_seed701.mat",
        "D2_timevary_delay_500Iter_jitter.mat",
        "D2_timevary_delay_500Iter_jitter_seed42"
    ],
}

DATASETS = {name: build_dataset(name, cfg, data_save_dir) for name, cfg in DATA_SETS.items()}
# --- main loop: per regime, per seed file ---
all_results = []  # will hold concatenated summaries with a 'seed_label' column

for regime_name, file_list in DLAG_FILES.items():
    print(f"\n=== Regime: {regime_name} ===")

    # keep only the matching dataset for this regime
    ALL_DATASETS = [DATASETS[regime_name]]

    for fname in file_list:
        mat_path = os.path.join(DLAG_fitted_path, fname)
        seed_label = os.path.splitext(fname)[0]  # e.g., "S0_linear_easy_500Iter_jitter_seed705"

        try:
            latents = load_dlag_latents_struct(mat_path)
            # unique output per run
            save_excel_path = os.path.join(
                eval_out_dir, f"{regime_name}__{seed_label}__DLAG_eval.xlsx"
            )
            print(f"[RUN] {regime_name} / {seed_label}")

            # evaluate (function you provided)
            summary_df = evaluate_oneModel_all_datasets_DLAG(
                latents=latents,
                ALL_DATASETS=ALL_DATASETS,
                save_excel_path=save_excel_path,
            )

            # tag with seed for downstream comparison and collect
            if isinstance(summary_df, pd.DataFrame) and len(summary_df) > 0:
                summary_df = summary_df.copy()
                summary_df["seed_label"] = seed_label
                all_results.append(summary_df)

        except Exception as e:
            print(f"[ERROR] {regime_name} / {seed_label}: {e}")

# --- optional: write a single combined Excel across ALL runs (summary_all + run_agg) ---
if all_results:
    combined = pd.concat(all_results, ignore_index=True)

    # Save the raw long-form summary CSV (unchanged behavior)
    combined_csv = os.path.join(eval_out_dir, "DLAG_all_runs_summary_longT.csv")
    combined.to_csv(combined_csv, index=False)
    print(f"\n[DONE] Wrote combined long-form summary to: {combined_csv}")

    # Recompute run_agg for the combined table (same logic as in your evaluate function)
    def _agg_run(g: pd.DataFrame):
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

        def first(col):
            s = g[col].dropna()
            return s.iloc[0] if len(s) else np.nan

        # carry through alignment if present
        for c in ["cca_shared12"]:
            if c in g.columns:
                out[c] = first(c)

        leak_terms = {
            "leak_shared_vs_gt_private":
                np.nanmean([g.get("cca_leak_s1_vs_gtp1", pd.Series(dtype=float)).mean(),
                            g.get("cca_leak_s2_vs_gtp2", pd.Series(dtype=float)).mean()]),
            "leak_private_vs_gt_shared":
                np.nanmean([g.get("cca_leak_p1_vs_gts1", pd.Series(dtype=float)).mean(),
                            g.get("cca_leak_p2_vs_gts2", pd.Series(dtype=float)).mean()]),
        }
        out.update(leak_terms)

        leak_pen = np.nanmean([out.get("leak_shared_vs_gt_private", np.nan),
                               out.get("leak_private_vs_gt_shared", np.nan)])
        out["composite_score"] = (
            0.45 * out["mean_cca_shared"]
          + 0.25 * out.get("cca_shared12", np.nan)
          - 0.10 * (leak_pen if not np.isnan(leak_pen) else 0.0)
        )
        return pd.Series(out)

    # Group by regime × variant × seed_label so you can compare seeds too
    group_keys = ["regime", "variant", "seed_label"]
    for k in group_keys:
        if k not in combined.columns:
            combined[k] = np.nan  # in case 'seed_label' is missing for some rows

    combined_run_agg = (combined
                        .groupby(group_keys, as_index=False)
                        .apply(_agg_run)
                        .reset_index(drop=True))

    # Optionally apply your derived columns function if available
    try:
        combined_run_agg = add_derived_columns_DLAG(combined_run_agg)
    except Exception as e:
        print(f"[INFO] Skipping add_derived_columns_DLAG for combined run_agg ({e}).")

    # Write a single Excel with both sheets
    combined_xlsx = os.path.join(eval_out_dir, "DLAG_all_runs_combinedT.xlsx")

    writer = None
    try:
        writer = pd.ExcelWriter(combined_xlsx, engine="xlsxwriter")
    except Exception:
        try:
            writer = pd.ExcelWriter(combined_xlsx, engine="openpyxl")
            print("[INFO] Using openpyxl engine for combined Excel output.")
        except Exception as e:
            print(f"[WARN] Excel engines unavailable ({e}). Writing CSVs instead.")
            base = os.path.splitext(combined_xlsx)[0]
            combined.to_csv(base + "_summary_all.csv", index=False)
            combined_run_agg.to_csv(base + "_run_agg.csv", index=False)
            print(f"[DONE] Wrote CSV summaries with base path: {base}")
        else:
            with writer:
                combined.to_excel(writer, index=False, sheet_name="summary_all")
                combined_run_agg.to_excel(writer, index=False, sheet_name="run_agg")
            print(f"[DONE] Wrote combined Excel to: {combined_xlsx}")
    else:
        with writer:
            combined.to_excel(writer, index=False, sheet_name="summary_all")
            combined_run_agg.to_excel(writer, index=False, sheet_name="run_agg")
        print(f"[DONE] Wrote combined Excel to: {combined_xlsx}")
else:
    print("\n[WARN] No results produced.")