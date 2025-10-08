# extra measure showing shift with rrespect to offstim in all latents
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]   # repo root
sys.path.insert(0, str(root / "src"))

import os
import torch
import pandas as pd
from evaluate_real import distribution_metrics_with_baseline

#similarly repeat this for STN stim
save_test_latent_dir = r"F:\comp_project\2region_models_SPIRE_dimSweep\test_latents_F"
excel_save_dir = r"F:\comp_project\2region_models_SPIRE_dimSweep\excels"


subject_list = [d for d in os.listdir(save_test_latent_dir) if os.path.isdir(os.path.join(save_test_latent_dir, d))]

# Dictionary to store subject: [list of GPi setting folders]
subject_settings = {}
all_dfs = []

for subj in subject_list:
    subj_path = os.path.join(save_test_latent_dir, subj)
    
    # Filter for folders that start with "GPi"
    gpi_folders = [
        f for f in os.listdir(subj_path)
        if os.path.isdir(os.path.join(subj_path, f)) and f.startswith("GPi")
    ]
    
    subject_settings[subj] = gpi_folders
    for setting in gpi_folders:
        path_latents = os.path.join(subj_path, setting)
        
        # Extract the side (first character after the underscore)
        try:
            side = setting.split("_")[1][0]  # 'L' from 'L12'
        except IndexError:
            side = "?"  # fallback if the format is unexpected
        
        print(f"Subject: {subj} | Setting: {setting} | Side: {side}")

        #load latents
        label_map = {0: "Off", 1: "85Hz", 2: "185Hz", 3: "250Hz"}
        shared_gpi = torch.load(os.path.join(path_latents, "shared_gpi_aligned.pt"))
        shared_stn = torch.load(os.path.join(path_latents, "shared_stn_aligned.pt"))
        private_gpi = torch.load(os.path.join(path_latents, "private_gpi.pt"))
        private_stn = torch.load(os.path.join(path_latents, "private_stn.pt"))

        print(f"Test latents Loaded")

        #calculate the measure for centroid shift for each latent type
        df = distribution_metrics_with_baseline(
            shared_gpi, shared_stn, private_gpi, private_stn,
            side, setting, subj,
            metrics=("energy","mmd"),
            sample_level="windows",   # or "timepoints"; windows is more conservative w.r.t. autocorr
            max_n=2000,
            rng_seed=42,
            baseline="holdout_mean",
            holdout_frac=0.5,         # 50% of Off as holdout set
            mmd_bandwidth=None        # None => median heuristic
        )

        all_dfs.append(df)

# Concatenate all and save
final_df = pd.concat(all_dfs, ignore_index=True)
final_df.to_excel(os.path.join(excel_save_dir, "dist_metrics_energy_mmd_SPIRE_F.xlsx"), index=False)
print("✅ Excel file saved!")

