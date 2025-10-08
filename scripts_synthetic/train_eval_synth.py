import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]   # repo root
sys.path.insert(0, str(root / "src"))

import numpy as np
import os
from scipy.io import savemat
import torch
from IPython.display import display

from data.synth_data import generate_synth_data
from data.data_loader import build_dataset_with_lag
from utils.training_utils import gen_ablation_variants, set_seed
from models.train import train_spire_synth
from evaluate_synth import evaluate_all_models_and_datasets

data_save_dir = r"F:\comp_project\synthecticData\dataT"#your target directory to save datasets
os.makedirs(data_save_dir, exist_ok=True)

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


# --- generate datasets & collect ---
ALL_DATASETS = []
for regime_name, kwargs in DATA_SETS.items():
    data = generate_synth_data(**kwargs)  

    # optional: save each regime
    if data_save_dir is not None:
        np.savez(os.path.join(data_save_dir, f"{regime_name}.npz"), **data) # save the datasets
        savemat(os.path.join(data_save_dir, f"{regime_name}.mat"), data) # save in .mat format to fit DLAG

    ALL_DATASETS.append({
        "regime": regime_name,
        "config": kwargs,   # store exactly what we passed in
        "data": data,       # full dict returned by generator
    })

# quick sanity check
print(len(ALL_DATASETS), [d["regime"] for d in ALL_DATASETS])

# --- train the base model and ablation variants and control variants ---
BASE = {
    "w_rec":1.0, "w_align":1.0, "w_orth":1.0,
    "w_align_reg":1.0,
    "w_cross":1.0, "w_self":1.0,"w_mapid":1.0
}
MODEL_VARIANTS_ABL = gen_ablation_variants(BASE)

# train the 4 extra control variants 
CTRL_VARIANTS = {
    "ctrl_no_var_guard": dict(BASE),

    # Identity aligner (bypass convs; mapper only)
    "ctrl_identity_aligner": dict(BASE),

    # No private ramp (α_p fixed to 1)
    "ctrl_no_private_ramp": dict(BASE),

    # No freeze window
    "ctrl_no_freeze": dict(BASE),   
}

#train the model variants on four seeds
seed_list = [701,702,703,704]

# Training hyperparams
num_epochs = 500
patience = 20
shared_dim = 3
private_dim = 3
LAGS= 3
model_save_dir = r"F:\comp_project\synthecticData\model_val_sweepT"

os.makedirs(model_save_dir, exist_ok=True)

# Loop over datasets and ablation variants
TRAINED_MODELS = []
for seed_model in seed_list:
    print(seed_model)
    for dataset in ALL_DATASETS:
        regime = dataset["regime"]
        data = dataset["data"]

        # unpack data
        region1_data, region2_data = data["region1"], data["region2"] #N,T,C

        # helper builds lag-augmented features and returns (N, C*lags+1, T-lags)
        reg1_R, reg2_R = build_dataset_with_lag(region1_data, region2_data, lags=LAGS)
        # print(np.shape(reg1_R))

        # --- Convert to tensors and permute to (N, T, C) ---
        reg1_tensor = torch.tensor(reg1_R, dtype=torch.float32).permute(0, 2, 1)
        reg2_tensor = torch.tensor(reg2_R, dtype=torch.float32).permute(0, 2, 1)

        # use all as train (adjust if you want a split) we follow what DLAG did in demo
        reg1_train, reg2_train = reg1_tensor, reg2_tensor
        reg1_test, reg2_test = reg1_tensor, reg2_tensor

        set_seed(seed_model)

        for variant_name, weights in MODEL_VARIANTS_ABL.items():
            run_name = f"{regime}_{variant_name}_E{num_epochs}_seed{seed_model}"
            model_save_path = os.path.join(model_save_dir, f"{run_name}.pt")

            print(f"Training {variant_name} on {regime}...")

            model, val_loader, device = train_spire_synth(
                reg1_train, reg2_train,
                shared_dim=shared_dim, private_dim=private_dim,
                num_epochs=num_epochs, patience=patience,
                run_name=run_name, model_save_path=model_save_path,
                **weights  # pass loss weights
            )

            TRAINED_MODELS.append({
                "regime": regime,
                "variant": variant_name,
                "model_path": model_save_path,
                "model": model,
                "seed":seed_model
            })

# Loop over datasets and control variants
for seed_model in seed_list:
    print(seed_model)
    for dataset in ALL_DATASETS:
        regime = dataset["regime"]
        data = dataset["data"]

        # unpack data
        region1_data, region2_data = data["region1"], data["region2"] #N,T,C

        # helper builds lag-augmented features and returns (N, C*lags+1, T-lags)
        reg1_R, reg2_R = build_dataset_with_lag(region1_data, region2_data, lags=LAGS)
        # print(np.shape(reg1_R))

        # --- Convert to tensors and permute to (N, T, C) ---
        reg1_tensor = torch.tensor(reg1_R, dtype=torch.float32).permute(0, 2, 1)
        reg2_tensor = torch.tensor(reg2_R, dtype=torch.float32).permute(0, 2, 1)

        # use all as train (adjust if you want a split) we follow what DLAG did in demo
        reg1_train, reg2_train = reg1_tensor, reg2_tensor
        reg1_test, reg2_test = reg1_tensor, reg2_tensor

        set_seed(seed_model)

        for variant_name, weights in CTRL_VARIANTS.items():
            run_name = f"{regime}_{variant_name}_E{num_epochs}_seed{seed_model}"
            model_save_path = os.path.join(model_save_dir, f"{run_name}.pt")

            # pick flags
            kwargs = {}
            if variant_name == "ctrl_identity_aligner":
                kwargs["identity_aligner"] = True
            elif variant_name == "ctrl_no_private_ramp":
                kwargs["disable_private_ramp"] = True
            elif variant_name == "ctrl_no_freeze":
                kwargs["FREEZE_BEG"] = None
                kwargs["FREEZE_END"] = None
            elif variant_name == "ctrl_no_var_guard":
                kwargs["var_guards"] = False

            print(f"Training control {variant_name} on {regime}...")

            model, val_loader, device = train_spire_synth(
                reg1_train, reg2_train,
                shared_dim=shared_dim, private_dim=private_dim,
                num_epochs=num_epochs, patience=patience,
                run_name=run_name, model_save_path=model_save_path,
                **kwargs,      # ⬅️ control flags
                **weights  # pass loss weights
            )

            TRAINED_MODELS.append({
                "regime": regime,
                "variant": variant_name,
                "model_path": model_save_path,
                "model": model,
                "seed":seed_model
            })


# now evaluate all models trained on all 3 datasets and all 4 seeds (function also saves latents for later evaluation or vizualization)
print(f"Evaluating ...")
results_df = evaluate_all_models_and_datasets(
    TRAINED_MODELS=TRAINED_MODELS,
    ALL_DATASETS=ALL_DATASETS,
    save_excel_path=r"F:\comp_project\synthecticData\evaluation\spire_synth_abl_ctrl_AllseedT.xlsx",
    device="cuda" if torch.cuda.is_available() else "cpu",
    lags=LAGS,
    latent_save_dir=r"F:\comp_project\synthecticData\SPIREdata"
)
display(results_df.head())
