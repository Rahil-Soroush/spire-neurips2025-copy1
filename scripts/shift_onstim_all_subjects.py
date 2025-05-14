import os
import torch
import pandas as pd
from src.analysis import calculate_pointwise_distribution_shift_sampled

#pointwise distribution shift

save_test_latent_dir = r"F:\comp_project\2region_models\test_latents_s3_p5"
excel_save_dir = r"F:\comp_project\2region_models\excels" 

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
        shared_gpi = torch.load(os.path.join(path_latents, "shared_gpi.pt"))
        shared_stn = torch.load(os.path.join(path_latents, "shared_stn.pt"))
        private_gpi = torch.load(os.path.join(path_latents, "private_gpi.pt"))
        private_stn = torch.load(os.path.join(path_latents, "private_stn.pt"))

        print(f"Test latents Loaded")

        #calculate the measure for pointwise ditribution shift for each latent type
        df = calculate_pointwise_distribution_shift_sampled(shared_gpi, shared_stn, private_gpi, private_stn, side, setting, subj)


        all_dfs.append(df)

# Concatenate all and save
final_df = pd.concat(all_dfs, ignore_index=True)
final_df.to_csv(os.path.join(excel_save_dir, "pointwise_distribution_shifts_500.csv"), index=False)
print("✅ Excel file saved!")
