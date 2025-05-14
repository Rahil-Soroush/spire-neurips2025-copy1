#classification

import os
import torch
import pandas as pd
from src.analysis import calculate_RF_accuracy

save_test_latent_dir = r"F:\comp_project\2region_models\test_latents_onstim_s3p5"
excel_save_dir = r"F:\comp_project\2region_models\excels_s3p5" 



subject_list = [d for d in os.listdir(save_test_latent_dir) if os.path.isdir(os.path.join(save_test_latent_dir, d))]


# Dictionary to store subject: [list of GPi setting folders]
subject_settings = {}
all_acc_dfs, all_imp_dfs = [], []

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
        shared_gpi = torch.load(os.path.join(path_latents, "shared_gpi.pt"))
        shared_stn = torch.load(os.path.join(path_latents, "shared_stn.pt"))
        private_gpi = torch.load(os.path.join(path_latents, "private_gpi.pt"))
        private_stn = torch.load(os.path.join(path_latents, "private_stn.pt"))

        print(f"Test latents Loaded")

        #calculate the measure for pointwise ditribution shift for each latent type
        df_acc, df_imp = calculate_RF_accuracy(shared_gpi, shared_stn, private_gpi, private_stn, side, setting, subj)
        all_acc_dfs.append(df_acc)
        all_imp_dfs.append(df_imp)


# Concatenate all and save
final_acc_df = pd.concat(all_acc_dfs, ignore_index=True)
final_imp_df = pd.concat(all_imp_dfs, ignore_index=True)
final_acc_df.to_excel(os.path.join(excel_save_dir, "RF_accuracy.xlsx"), index=False)
final_imp_df.to_excel(os.path.join(excel_save_dir, "RF_importance.xlsx"), index=False)

print("✅ Excel file saved!")
