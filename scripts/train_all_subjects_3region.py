import os

import torch

from src.data.data_loader import load_paired_segments_multiRegion_with_filtering, build_dataset_with_lag_multiRegion
from src.train import train_spire_3Region

subject_list = ["s508", "s513", "s515", "s518","s520"]  # subject IDs with vo and va which have only one folder for baseline
baseline_folder_list = ["-11", "-14", "-9", "-21","-13"]  # corresponding baseline folders

base_dir = r"D:\comp_project\LPF_Data\imagingContacts"  # Base directory path
data_save_dir = r"D:\comp_project\Off_tensor_Data_R"
model_save_dir = r"D:\comp_project\3region_models"

for subj, baseline_folder in zip(subject_list, baseline_folder_list):
    print(f"\n🚀 Processing subject {subj}...")

    # === 1. Build the path
    path_R = os.path.join(base_dir, subj, f"Offstim_R_{baseline_folder}")

    # === 2. Load and preprocess data
    gpi_segs_R1, vo_segs_R1, stn_segs_R1, va_segs_R1, fs = load_paired_segments_multiRegion_with_filtering(
        path_R, segment_length=0.5, channel_idx=0, cutoff=50, order=11
    )
    gpi_R, vo_R, stn_R, va_R = build_dataset_with_lag_multiRegion(gpi_segs_R1, vo_segs_R1, stn_segs_R1, va_segs_R1, lags=3)

    # Convert to tensors
    vo_R = torch.tensor(vo_R, dtype=torch.float32)
    va_R = torch.tensor(va_R, dtype=torch.float32)
    thal_R = torch.cat([vo_R, va_R], dim=1)

    gpi_tensor = torch.tensor(gpi_R, dtype=torch.float32)
    stn_tensor = torch.tensor(stn_R, dtype=torch.float32)
    thal_tensor = torch.tensor(thal_R, dtype=torch.float32)

    # Permute to (N, T, Channels)
    gpi_tensor = gpi_tensor.permute(0, 2, 1)
    stn_tensor = stn_tensor.permute(0, 2, 1)
    thal_tensor = thal_tensor.permute(0, 2, 1)

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

    thal_train_off = thal_tensor[train_indices]
    thal_test_off = thal_tensor[test_indices]

    # === 4. Save train/test sets
    save_dir = os.path.join(data_save_dir, subj)
    os.makedirs(save_dir, exist_ok=True)

    torch.save(gpi_train_off, os.path.join(save_dir, "gpi_train_off.pt"))
    torch.save(gpi_test_off, os.path.join(save_dir, "gpi_test_off.pt"))
    torch.save(stn_train_off, os.path.join(save_dir, "stn_train_off.pt"))
    torch.save(stn_test_off, os.path.join(save_dir, "stn_test_off.pt"))
    torch.save(thal_train_off, os.path.join(save_dir, "thal_train_off.pt"))
    torch.save(thal_test_off, os.path.join(save_dir, "thal_test_off.pt"))

    print(f"✅ Saved train/test tensors for {subj}.")

    # === 5. Train model
    model_save_path = os.path.join(model_save_dir, subj, f"model_P20_sD4_pD6_R.pt")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    model, val_loader, device = train_spire_3Region(
        gpi_train_off, stn_train_off, thal_train_off,
        shared_dim=4, private_dim=6,
        run_name=f"subject_{subj}_R_shD4_prD6",
        model_save_path=model_save_path
    )

    print(f"✅ Finished training and saving model for {subj}.")

print("\n🎯 All subjects processed successfully!")
