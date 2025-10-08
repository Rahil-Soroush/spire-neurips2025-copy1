import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]   # repo root
sys.path.insert(0, str(root / "src"))

import os
import torch
from data.data_loader import build_dataset_with_lag, load_paired_segments_with_filtering
from utils.training_utils import sched_real
from models.train import train_spire_real

base_dir = r"F:\comp_project\LPF_Data\imagingContacts"  # Base directory path
model_save_dir = r"F:\comp_project\2region_models_SPIRE_dimSweep"
os.makedirs(model_save_dir, exist_ok=True)
results_dir = r"F:\comp_project\2region_models_SPIRE_dimSweep\excels"
os.makedirs(results_dir, exist_ok=True)

Real_weight_CFG = dict( #not used really
    w_rec=1.0, w_align=1.0, w_orth=1.0,
    w_self=1.0, w_cross=1.0,
    w_mapid=1.0, w_align_reg=1.0
)

sd_list = (3,4,5)
pd_list = (2,3,4)
num_epochs = 200

data_save_dir = r"F:\comp_project\Off_tensor_Data_R"
subjects = ["s515"]  # right side
baseline_folder_list = ["-11"]  # corresponding baseline folders
run_prefix="SPIRE_final_R"

for subj, baseline_folder in zip(subjects, baseline_folder_list):
    print(f"\n🚀 Processing subject {subj}...")

    # === 1. Build the path
    path_R = os.path.join(base_dir, subj, f"Offstim_R_{baseline_folder}")#####side

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


    for sd in sd_list:
        for pdim in pd_list:
            run_name = f"{run_prefix}_{subj}_sd{sd}_pd{pdim}_map"
            model_path = os.path.join(model_save_dir, subj, f"{run_name}.pt")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            print(f"\n=== {run_name} ===")
            _, _, _ = train_spire_real(
                gpi_train_off, stn_train_off,
                shared_dim=sd, private_dim=pdim,
                num_epochs=num_epochs,
                run_name=run_name,
                model_save_path=model_path,
                weight_schedule=sched_real,
                dropout_prob=0.2,
                FREEZE_BEG=None, FREEZE_END = None,
                **Real_weight_CFG
            )

#_____left side
data_save_dir = r"F:\comp_project\Off_tensor_Data_L_"
subjects = ["s508"]  # left side
baseline_folder_list = ["-11"]  # corresponding baseline folders
run_prefix="SPIRE_final_L"

for subj, baseline_folder in zip(subjects, baseline_folder_list):
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


    for sd in sd_list:
        for pdim in pd_list:
            run_name = f"{run_prefix}_{subj}_sd{sd}_pd{pdim}_map"
            model_path = os.path.join(model_save_dir, subj, f"{run_name}.pt")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            print(f"\n=== {run_name} ===")
            _, _, _ = train_spire_real(
                gpi_train_off, stn_train_off,
                shared_dim=sd, private_dim=pdim,
                num_epochs=num_epochs,
                run_name=run_name,
                model_save_path=model_path,
                weight_schedule=sched_real,
                dropout_prob=0.2,
                FREEZE_BEG=None, FREEZE_END = None,
                **Real_weight_CFG
            )