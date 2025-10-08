import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]   # repo root
sys.path.insert(0, str(root / "src"))

import os
from data.synth_data import load_saved_dataset
from visualization.synth_visualizer import plot_joint_latents_figure

#load all datasets
data_save_dir = r"F:\comp_project\synthecticData\dataT"#your target directory to save datasets

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
ALL_DATASETS = []
for regime_name, config in DATA_SETS.items():
    ds = load_saved_dataset(regime_name, config, data_save_dir)
    ALL_DATASETS.append(ds)

#generate figure A.1
figure_save_dir = r"F:\comp_project\synthecticData\figuresT"
os.makedirs(figure_save_dir, exist_ok=True)

plot_joint_latents_figure(ALL_DATASETS, trial_idx=0, secs=0.6,
                          outpath=os.path.join(figure_save_dir, "synthetic_joint_overview.pdf"),
                          latent_ylim=(-2.5, 2.0))