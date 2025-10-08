#after choosing best model for eachs ubject we quantify the performance of model using CCA and reconstruction MSE
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]   # repo root
sys.path.insert(0, str(root / "src"))

import os
import torch
import pandas as pd
from evaluate_real import get_measuresAll_df_per_sample
from models.spire_model import SPIREAutoencoder, LatentEncoder, LatentDecoder, ConvAlign1D

model_save_dir = r"F:\comp_project\2region_models_SPIRE_dimSweep"
excel_save_dir = r"F:\comp_project\2region_models_SPIRE_dimSweep\excels"


subject_dims_R = {  # subject-specific (sd, pdim)
    "s508": (3,2),
    "s514": (3,3),
    "s515": (3,4),
    "s517": (3,2),
    "s519": (5,2),
    "s520": (3,4),
    "s521": (5, 2),
    "s523": (5,4),
}
subject_dims_L = {  # subject-specific (sd, pdim)
    "s508": (3,4),
    "s513": (5,3),
    "s514": (5,2),
    "s515": (5,4),
    "s518": (3,2),
    "s519": (3,4),
    "s520": (4,3),
    "s521": (5, 2),
    "s523": (5,4),
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# #______________right side
# data_save_dir = r"F:\comp_project\Off_tensor_Data_R"
# subjects = ["s508","s514", "s515","s517","s519","s520","s521","s523"]  # right side
# run_prefix="SPIRE_final_R"
# side="R"


#____________Left side
subject_list = ["s508", "s513","s514", "s515","s518","s519","s520","s521","s523"]  # left side
data_save_dir = r"F:\comp_project\Off_tensor_Data_L" #####side
run_prefix="SPIRE_final_L"
side="L"


if side == "R":
    subject_dims = subject_dims_R
elif side=="L":
    subject_dims = subject_dims_L

all_dfs_recon = []  # collect all subject results
all_dfs_simil = []  # collect all subject results

for subj in subject_list:
    print(f"\n Processing subject {subj}...")
    
    sd, pdim = subject_dims[subj]
    # 1. Load the test tensors
    Off_test_data_dir = os.path.join(data_save_dir, subj)
    gpi_test_off = torch.load(os.path.join(Off_test_data_dir, "gpi_test_off.pt"))
    stn_test_off = torch.load(os.path.join(Off_test_data_dir, "stn_test_off.pt"))

    print(f"Loaded: {gpi_test_off.shape}, {stn_test_off.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = f"{run_prefix}_{subj}_sd{sd}_pd{pdim}_R2_map"
    model_save_path = os.path.join(model_save_dir,subj, f"{run_name}_bundle.pth")
    checkpoint = torch.load(model_save_path)
    model = checkpoint['model']
    model.eval()
    print(f"model Loaded")

    # 4. Run and collect metrics
    df_recon, df_simil = get_measuresAll_df_per_sample(model, gpi_test_off, stn_test_off, device=device, side=side, subject_id=subj)

    all_dfs_recon.append(df_recon)
    all_dfs_simil.append(df_simil)

# Concatenate all and save
final_df_recon = pd.concat(all_dfs_recon, ignore_index=True)
final_df_recon.to_excel(os.path.join(excel_save_dir, f"MSE_results{run_prefix}_{side}.xlsx"), index=False)
final_df_simil = pd.concat(all_dfs_simil, ignore_index=True)
final_df_simil.to_excel(os.path.join(excel_save_dir, f"simil_results{run_prefix}_{side}.xlsx"), index=False)
print("✅ Excel file saved!")