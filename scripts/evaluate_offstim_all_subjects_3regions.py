import os
import torch
from src.evaluate import get_mse_r2_df_per_sample_3Regions
import pandas as pd


subject_list = ["s508","s515","s519","s520","s521","s523"]  # right side


data_save_dir = r"F:\comp_project\Off_tensor_Data_R" #####side
model_save_dir = r"F:\comp_project\3region_models"
excel_save_dir = r"F:\comp_project\3region_models\excels_s4p6"

side="R"

all_dfs = []  # collect all subject results

for subj in subject_list:
    print(f"\n Processing subject {subj}...")

    # 1. Load the test tensors
    Off_test_data_dir = os.path.join(data_save_dir, subj)
    gpi_test_off = torch.load(os.path.join(Off_test_data_dir, "gpi_test_off.pt"))
    stn_test_off = torch.load(os.path.join(Off_test_data_dir, "stn_test_off.pt"))
    thal_test_off = torch.load(os.path.join(Off_test_data_dir, "thal_test_off.pt"))

    print(f"Loaded: {gpi_test_off.shape}, {stn_test_off.shape},{thal_test_off.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_save_path = os.path.join(model_save_dir, subj, f"model_P20_sD4_pD6_R.pt") ########side
    checkpoint = torch.load(model_save_path)

    model = checkpoint['model']
    model.eval()

    print(f"model Loaded")

#     # 4. Run and collect metrics
    df = get_mse_r2_df_per_sample_3Regions(model,  gpi_test_off, stn_test_off, thal_test_off,device=device,side = "R", subject_id=subj)

    all_dfs.append(df)

# Concatenate all and save
final_df = pd.concat(all_dfs, ignore_index=True)
final_df.to_excel(os.path.join(excel_save_dir, "MSE_results_sD4_pD6_R.xlsx"), index=False)####L
print("✅ Excel file saved!")