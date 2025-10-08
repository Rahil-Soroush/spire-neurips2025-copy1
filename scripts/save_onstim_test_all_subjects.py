#next for evaluating the effect of stim, we need to save latents when given onstim unseen data
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]   # repo root
sys.path.insert(0, str(root / "src"))

import os
from data.data_loader import load_paired_segments_onstim_with_filtering, build_dataset_with_lag
import torch
from evaluate_real import extract_latents_by_condition
from models.spire_model import SPIREAutoencoder, LatentEncoder, LatentDecoder, ConvAlign1D

##########______________GPi stim, similarly repeat for STN stim
base_dir = r"F:\comp_project\LPF_Data\imagingContacts"  # Base directory path
model_save_dir = r"F:\comp_project\2region_models_SPIRE_dimSweep"
save_test_latent_dir = r"F:\comp_project\2region_models_SPIRE_dimSweep\test_latents_F"
os.makedirs(save_test_latent_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

subject_list = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
# Dictionary to store subject: [list of GPi setting folders]
subject_settings = {}

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

LAGS= 3
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
        
        # --- pick dims by subject & side ---
        if side == "R":
            dims_map = subject_dims_R
        elif side == "L":
            dims_map = subject_dims_L
        else:
            print(f"[WARN] Unknown side '{side}' in setting '{setting}'. Skipping.")
            continue
        if subj not in dims_map:
            print(f"[WARN] No dims configured for subject {subj} on side {side}. Skipping.")
            continue
        sd, pdim = dims_map[subj]
        print(f"Subject: {subj} | Setting: {setting} | Side: {side}")

        #if for R or L?
        freq = 85  # or 185 or 250
        gpi_segs_on, stn_segs_on, fs_on = load_paired_segments_onstim_with_filtering(path_onstim, freq, segment_length=0.5,channel_idx=0, cutoff=50,order=11)
        X_on, y_on = build_dataset_with_lag(gpi_segs_on, stn_segs_on, lags=LAGS)
        X_tensor_85 = torch.tensor(X_on, dtype=torch.float32).permute(0, 2, 1)
        y_tensor_85 = torch.tensor(y_on, dtype=torch.float32).permute(0, 2, 1)

        freq = 185  # or 185 or 250
        gpi_segs_on, stn_segs_on, fs_on = load_paired_segments_onstim_with_filtering(path_onstim, freq, segment_length=0.5,channel_idx=0, cutoff=50,order=11)
        X_on, y_on = build_dataset_with_lag(gpi_segs_on, stn_segs_on, lags=LAGS)
        X_tensor_185 = torch.tensor(X_on, dtype=torch.float32).permute(0, 2, 1)
        y_tensor_185 = torch.tensor(y_on, dtype=torch.float32).permute(0, 2, 1)

        freq = 250  # or 185 or 250
        gpi_segs_on, stn_segs_on, fs_on = load_paired_segments_onstim_with_filtering(path_onstim, freq, segment_length=0.5,channel_idx=0, cutoff=50,order=11)
        X_on, y_on = build_dataset_with_lag(gpi_segs_on, stn_segs_on, lags=LAGS)

        X_tensor_250 = torch.tensor(X_on, dtype=torch.float32).permute(0, 2, 1) #N, T, ch
        y_tensor_250 = torch.tensor(y_on, dtype=torch.float32).permute(0, 2, 1)


        if side == "R":
            off_dir = r"F:\comp_project\Off_tensor_Data_R" #####side
            run_prefix = "SPIRE_final_R" ####side
            run_name = f"{run_prefix}_{subj}_sd{sd}_pd{pdim}_map"
            model_save_path = os.path.join(model_save_dir,subj, f"{run_name}_bundle.pth")
        elif side == "L":
            off_dir = r"F:\comp_project\Off_tensor_Data_L" #####side
            run_prefix = "SPIRE_final_L" ####side
            run_name = f"{run_prefix}_{subj}_sd{sd}_pd{pdim}_map"
            model_save_path = os.path.join(model_save_dir,subj, f"{run_name}_bundle.pth")
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
        
        checkpoint = torch.load(model_save_path)
        model = checkpoint['model']
        model.eval()

        print(f"model Loaded")


        #extract latents
        label_map = {0: "Off", 1: "85Hz", 2: "185Hz", 3: "250Hz"}

        shared_gpi, shared_stn, private_gpi, private_stn,shared_gpi_aligned,shared_stn_aligned = extract_latents_by_condition(
            model, X_test_all, Y_test_all, labels_test_all, device, label_map
        ) #shape for each condition: N, T, dim

        save_test_latent_dir
        save_dir = os.path.join(save_test_latent_dir, subj, setting)
        os.makedirs(save_dir, exist_ok=True)

        torch.save(shared_gpi, os.path.join(save_dir, "shared_gpi.pt"))
        torch.save(shared_stn, os.path.join(save_dir, "shared_stn.pt"))
        torch.save(private_gpi, os.path.join(save_dir, "private_gpi.pt"))
        torch.save(private_stn, os.path.join(save_dir, "private_stn.pt"))
        torch.save(shared_gpi_aligned, os.path.join(save_dir, "shared_gpi_aligned.pt"))
        torch.save(shared_stn_aligned, os.path.join(save_dir, "shared_stn_aligned.pt"))
        
print("✅ All subjects done!")


##############STN stim
#subjects with STN stim: 513,517,518,523 and the ones with no 250 Hz: 508, 514, 519, 521
subject_list = [ "s513","s517","s518","s523"]
# subject_list = [ "s508","s514","s519","s521"] #with no 250 Hz

# Dictionary to store subject: [list of GPi setting folders]
subject_settings = {}
for subj in subject_list:
    subj_path = os.path.join(base_dir, subj)
    
    # Filter for folders that start with "GPi"
    stn_folders = [
        f for f in os.listdir(subj_path)
        if os.path.isdir(os.path.join(subj_path, f)) and f.startswith("VoSTN")
    ]
    
    subject_settings[subj] = stn_folders
    for setting in stn_folders:
        path_onstim = os.path.join(subj_path, setting)
        
        # Extract the side (first character after the underscore)
        try:
            side = setting.split("_")[1][0]  # 'L' from 'L12'
        except IndexError:
            side = "?"  # fallback if the format is unexpected
        # --- pick dims by subject & side ---
        if side == "R":
            dims_map = subject_dims_R
        elif side == "L":
            dims_map = subject_dims_L
        else:
            print(f"[WARN] Unknown side '{side}' in setting '{setting}'. Skipping.")
            continue
        if subj not in dims_map:
            print(f"[WARN] No dims configured for subject {subj} on side {side}. Skipping.")
            continue
        sd, pdim = dims_map[subj]
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
            run_prefix = "SPIRE_final_R" ####side
            run_name = f"{run_prefix}_{subj}_sd{sd}_pd{pdim}_map"
            model_save_path = os.path.join(model_save_dir,subj, f"{run_name}_bundle.pth")

        elif side == "L":
            off_dir = r"F:\comp_project\Off_tensor_Data_L" #####side
            run_prefix = "SPIRE_final_L" ####side
            run_name = f"{run_prefix}_{subj}_sd{sd}_pd{pdim}_map"
            model_save_path = os.path.join(model_save_dir,subj, f"{run_name}_bundle.pth")
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

        shared_gpi, shared_stn, private_gpi, private_stn,shared_gpi_aligned,shared_stn_aligned = extract_latents_by_condition(
            model, X_test_all, Y_test_all, labels_test_all, device, label_map
        ) #shape for each condition: N, T, dim

        save_test_latent_dir
        save_dir = os.path.join(save_test_latent_dir, subj, setting)
        os.makedirs(save_dir, exist_ok=True)

        torch.save(shared_gpi, os.path.join(save_dir, "shared_gpi.pt"))
        torch.save(shared_stn, os.path.join(save_dir, "shared_stn.pt"))
        torch.save(private_gpi, os.path.join(save_dir, "private_gpi.pt"))
        torch.save(private_stn, os.path.join(save_dir, "private_stn.pt"))
        torch.save(shared_gpi_aligned, os.path.join(save_dir, "shared_gpi_aligned.pt"))
        torch.save(shared_stn_aligned, os.path.join(save_dir, "shared_stn_aligned.pt"))
        
print("✅ All subjects done!")



# repeat for subjects that don't have 250 Hz stim
subject_list = [ "s508","s514","s519","s521"] #with no 250 Hz

# Dictionary to store subject: [list of GPi setting folders]
subject_settings = {}

for subj in subject_list:
    subj_path = os.path.join(base_dir, subj)
    
    # Filter for folders that start with "GPi"
    stn_folders = [
        f for f in os.listdir(subj_path)
        if os.path.isdir(os.path.join(subj_path, f)) and f.startswith("VoSTN")
    ]
    
    subject_settings[subj] = stn_folders
    for setting in stn_folders:
        path_onstim = os.path.join(subj_path, setting)
        
        # Extract the side (first character after the underscore)
        try:
            side = setting.split("_")[1][0]  # 'L' from 'L12'
        except IndexError:
            side = "?"  # fallback if the format is unexpected
        # --- pick dims by subject & side ---
        if side == "R":
            dims_map = subject_dims_R
        elif side == "L":
            dims_map = subject_dims_L
        else:
            print(f"[WARN] Unknown side '{side}' in setting '{setting}'. Skipping.")
            continue
        if subj not in dims_map:
            print(f"[WARN] No dims configured for subject {subj} on side {side}. Skipping.")
            continue
        sd, pdim = dims_map[subj]
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


        if side == "R":
            off_dir = r"F:\comp_project\Off_tensor_Data_R" #####side
            run_prefix = "SPIRE_final_R" ####side
            run_name = f"{run_prefix}_{subj}_sd{sd}_pd{pdim}_map"
            model_save_path = os.path.join(model_save_dir,subj, f"{run_name}_bundle.pth")
        elif side == "L":
            off_dir = r"F:\comp_project\Off_tensor_Data_L" #####side
            run_prefix = "SPIRE_final_L" ####side
            run_name = f"{run_prefix}_{subj}_sd{sd}_pd{pdim}_map"
            model_save_path = os.path.join(model_save_dir,subj, f"{run_name}_bundle.pth")
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
        X_test_all = torch.cat([X_test_off_sub, X_tensor_85, X_tensor_185], dim=0)
        Y_test_all = torch.cat([y_test_off_sub, y_tensor_85, y_tensor_185], dim=0)

        labels_test_all = torch.cat([
            torch.zeros(len(X_test_off_sub)),             # 0 = Off
            torch.ones(len(X_tensor_85)),             # 1 = 85Hz
            2 * torch.ones(len(X_tensor_185))        # 2 = 185Hz
        ]).long()

        print("all GPi test shape:", X_test_all.shape)

        #load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_save_path)
        model = checkpoint['model']
        model.eval()

        print(f"model Loaded")


        #extract latents
        label_map = {0: "Off", 1: "85Hz", 2: "185Hz"}

        shared_gpi, shared_stn, private_gpi, private_stn,shared_gpi_aligned,shared_stn_aligned = extract_latents_by_condition(
            model, X_test_all, Y_test_all, labels_test_all, device, label_map
        ) #shape for each condition: N, T, dim

        save_test_latent_dir
        save_dir = os.path.join(save_test_latent_dir, subj, setting)
        os.makedirs(save_dir, exist_ok=True)

        torch.save(shared_gpi, os.path.join(save_dir, "shared_gpi.pt"))
        torch.save(shared_stn, os.path.join(save_dir, "shared_stn.pt"))
        torch.save(private_gpi, os.path.join(save_dir, "private_gpi.pt"))
        torch.save(private_stn, os.path.join(save_dir, "private_stn.pt"))
        torch.save(shared_gpi_aligned, os.path.join(save_dir, "shared_gpi_aligned.pt"))
        torch.save(shared_stn_aligned, os.path.join(save_dir, "shared_stn_aligned.pt"))
        
print("✅ All subjects done!")

