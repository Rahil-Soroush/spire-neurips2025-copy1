import os
import torch
from src.data.data_loader import load_paired_segments_onstim_with_filtering, build_dataset_with_lag
from src.analysis import extract_latents_by_condition

#GPi stim
base_dir = r"F:\comp_project\LPF_Data\imagingContacts"  # Base directory path
model_save_dir = r"F:\comp_project\2region_models"
save_test_latent_dir = r"F:\comp_project\2region_models\test_latents_onstim_s3p5"

subject_list = [ ...]

# Dictionary to store subject: [list of GPi setting folders]
subject_settings = {}


for subj in subject_list:
    subj_path = os.path.join(base_dir, subj)
    
    # Filter for folders that start with "GPi"
    gpi_folders = [
        f for f in os.listdir(subj_path)
        if os.path.isdir(os.path.join(subj_path, f)) and f.startswith("GPi")
    ]
    
    subject_settings[subj] = gpi_folders
    for setting in gpi_folders:
        path_onstim = os.path.join(subj_path, setting)
        
        # Extract the side (first character after the underscore)
        try:
            side = setting.split("_")[1][0]  # 'L' from 'L12'
        except IndexError:
            side = "?"  # fallback if the format is unexpected
        
        print(f"Subject: {subj} | Setting: {setting} | Side: {side}")

        #if for R or L?
        freq = 85  # or 185 or 250
        gpi_segs_on, stn_segs_on, fs_on = load_paired_segments_onstim_with_filtering(path_onstim, freq, segment_length=0.5,channel_idx=0, cutoff=50,order=11)
        X_on, y_on = build_dataset_with_lag(gpi_segs_on, stn_segs_on, lags=3)
        X_tensor_85 = torch.tensor(X_on, dtype=torch.float32).permute(0, 2, 1)
        y_tensor_85 = torch.tensor(y_on, dtype=torch.float32).permute(0, 2, 1)

        freq = 185  # or 185 or 250
        gpi_segs_on, stn_segs_on, fs_on = load_paired_segments_onstim_with_filtering(path_onstim, freq, segment_length=0.5,channel_idx=0, cutoff=50,order=11)
        X_on, y_on = build_dataset_with_lag(gpi_segs_on, stn_segs_on, lags=3)
        X_tensor_185 = torch.tensor(X_on, dtype=torch.float32).permute(0, 2, 1)
        y_tensor_185 = torch.tensor(y_on, dtype=torch.float32).permute(0, 2, 1)

        freq = 250  # or 185 or 250
        gpi_segs_on, stn_segs_on, fs_on = load_paired_segments_onstim_with_filtering(path_onstim, freq, segment_length=0.5,channel_idx=0, cutoff=50,order=11)
        X_on, y_on = build_dataset_with_lag(gpi_segs_on, stn_segs_on, lags=3)

        X_tensor_250 = torch.tensor(X_on, dtype=torch.float32).permute(0, 2, 1) #N, T, ch
        y_tensor_250 = torch.tensor(y_on, dtype=torch.float32).permute(0, 2, 1)

        if side == "R":
            off_dir = r"F:\comp_project\Off_tensor_Data_R" #####side
            model_save_path = os.path.join(model_save_dir, subj, f"model_P20_sD3_pD5_R.pt") ########side
        elif side == "L":
            off_dir = r"F:\comp_project\Off_tensor_Data_L" #####side
            model_save_path = os.path.join(model_save_dir, subj, f"model_P20_sD3_pD5_L.pt") ########side
        else: 
            print(f"Unknown side '{side}' in setting '{setting}'")

        Off_test_data_dir = os.path.join(off_dir, subj)
        gpi_test_off = torch.load(os.path.join(Off_test_data_dir, "gpi_test_off.pt"))
        stn_test_off = torch.load(os.path.join(Off_test_data_dir, "stn_test_off.pt"))

        print(f"Loaded: {gpi_test_off.shape}, {stn_test_off.shape}")

        # Number of Off-stim samples to match approximate number of On-stim trials
        n_off = 35

        # Randomly sample n_off segments from X_test_off
        indices = torch.randperm(gpi_test_off.size(0))[:n_off]
        X_test_off_sub = gpi_test_off[indices]
        y_test_off_sub = stn_test_off[indices]

        # Combine balanced test set
        X_test_all = torch.cat([X_test_off_sub, X_tensor_85, X_tensor_185, X_tensor_250], dim=0)
        Y_test_all = torch.cat([y_test_off_sub, y_tensor_85, y_tensor_185, y_tensor_250], dim=0)

        labels_test_all = torch.cat([
            torch.zeros(len(X_test_off_sub)),             # 0 = Off
            torch.ones(len(X_tensor_85)),             # 1 = 85Hz
            2 * torch.ones(len(X_tensor_185)),        # 2 = 185Hz
            3 * torch.ones(len(X_tensor_250))         # 3 = 250Hz
        ]).long()

        print("all GPi test shape:", X_test_all.shape)

        #load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_save_path)
        model = checkpoint['model']
        model.eval()

        print(f"model Loaded")


        #extract latents
        label_map = {0: "Off", 1: "85Hz", 2: "185Hz", 3: "250Hz"}

        shared_gpi, shared_stn, private_gpi, private_stn = extract_latents_by_condition(
            model, X_test_all, Y_test_all, labels_test_all, device, label_map
        ) #shape for each condition: N, T, dim

        save_test_latent_dir
        save_dir = os.path.join(save_test_latent_dir, subj, setting)
        os.makedirs(save_dir, exist_ok=True)

        torch.save(shared_gpi, os.path.join(save_dir, "shared_gpi.pt"))
        torch.save(shared_stn, os.path.join(save_dir, "shared_stn.pt"))
        torch.save(private_gpi, os.path.join(save_dir, "private_gpi.pt"))
        torch.save(private_stn, os.path.join(save_dir, "private_stn.pt"))
        
print("✅ All subjects done!")
