import os

import torch

from src.data.data_loader import load_paired_segments_with_filtering, build_dataset_with_lag
from src.train import train_spire

subject_list = ["s508", "s513", "s515","s518","s520","s521","s523"]  # left side
baseline_folder_list = ["-11","-14","-9","-21" ,"-13","-9","-12"]  # corresponding baseline folders
# subject_list = ["s515","s517","s520","s521","s523"]  # right side with only one baseline folder
# baseline_folder_list = ["-9","-17" ,"-13","-9","-12"]  # corresponding baseline folders

base_dir = r"D:\comp_project\LPF_Data\imagingContacts"  # Base directory path
data_save_dir = r"D:\comp_project\2lag\Off_tensor_Data_L"#############side
model_save_dir = r"D:\comp_project\2lag\2region_models"

for subj, baseline_folder in zip(subject_list, baseline_folder_list):
    print(f"\n🚀 Processing subject {subj}...")

    # === 1. Build the path
    path_R = os.path.join(base_dir, subj, f"Offstim_L_{baseline_folder}")#####side

    # === 2. Load and preprocess data
    gpi_segs_R1, stn_segs_R1, fs = load_paired_segments_with_filtering(
        path_R, segment_length=0.5, channel_idx=0, cutoff=50, order=11
    )
    gpi_R, stn_R = build_dataset_with_lag(gpi_segs_R1, stn_segs_R1, lags=2)

    # Convert to tensors

    gpi_tensor = torch.tensor(gpi_R, dtype=torch.float32)
    stn_tensor = torch.tensor(stn_R, dtype=torch.float32)
   
    # Permute to (N, T, Channels)
    gpi_tensor = gpi_tensor.permute(0, 2, 1)
    stn_tensor = stn_tensor.permute(0, 2, 1)

    # === 3. Train/test split
    num_samples = gpi_tensor.shape[0]
    indices = torch.randperm(num_samples)

    train_size = int(0.8 * num_samples)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    gpi_train_off = gpi_tensor[train_indices]
    gpi_test_off = gpi_tensor[test_indices]

    stn_train_off = stn_tensor[train_indices]
    stn_test_off = stn_tensor[test_indices]

    # === 4. Save train/test sets
    save_dir = os.path.join(data_save_dir, subj)
    os.makedirs(save_dir, exist_ok=True)

    torch.save(gpi_train_off, os.path.join(save_dir, "gpi_train_off.pt"))
    torch.save(gpi_test_off, os.path.join(save_dir, "gpi_test_off.pt"))
    torch.save(stn_train_off, os.path.join(save_dir, "stn_train_off.pt"))
    torch.save(stn_test_off, os.path.join(save_dir, "stn_test_off.pt"))

    print(f"✅ Saved train/test tensors for {subj}.")

    # === 5. Train model
    model_save_path = os.path.join(model_save_dir, subj, f"spire_model_P20_sD3_pD3_L.pt")####side
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    model, val_loader, device = train_spire(
        gpi_train_off, stn_train_off,
        shared_dim=3, private_dim=5,
        run_name=f"subject_{subj}_L_shD3_prD5",####side
        model_save_path=model_save_path
    )

    print(f"✅ Finished training and saving model for {subj}.")

print("\n🎯 All subjects processed successfully!")