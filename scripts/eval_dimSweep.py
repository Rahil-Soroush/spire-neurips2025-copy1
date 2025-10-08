#after running the model for all subjects on all dimesnions here we estimate the variance explained to help us choose the best dim for each subject
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]   # repo root
sys.path.insert(0, str(root / "src"))

import os
import torch
import pandas as pd
import numpy as np
from evaluate_real import as_3d, extract_latents_from_test_set_align, to_np, cv_decode_fve, safe_clip
from models.spire_model import SPIREAutoencoder, LatentEncoder, LatentDecoder, ConvAlign1D

model_save_dir = r"F:\comp_project\2region_models_SPIRE_dimSweep"
excel_save_dir = r"F:\comp_project\2region_models_SPIRE_dimSweep\excels"

shared_dims  = [3, 4, 5]
private_dims = [2, 3, 4]
K_FOLDS = 5
RIDGE_ALPHAS = (1e-3, 1e-2, 1e-1, 1, 10)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#______________right side
data_save_dir = r"F:\comp_project\Off_tensor_Data_R"
subjects = ["s508","s514", "s515","s517","s519","s520","s521","s523"]  # right side
run_prefix="SPIRE_final_R"
side="R"

all_rows = []
for subj in subjects:
    print(f"\nProcessing subject {subj}...")

    # 1) Load observed test tensors (targets)
    off_dir = os.path.join(data_save_dir, subj)
    gpi_test_off = torch.load(os.path.join(off_dir, "gpi_test_off.pt"))   # expected (N, T, Cg)
    stn_test_off = torch.load(os.path.join(off_dir, "stn_test_off.pt"))   # expected (N, T, Cs)
    print(f"Loaded X: GPi {tuple(gpi_test_off.shape)}, STN {tuple(stn_test_off.shape)}")
    Y_gpi  = as_3d(gpi_test_off)  # (N, T, Cg)
    Y_stn  = as_3d(stn_test_off)  # (N, T, Cs)

    # Sweep the (sd, pdim) grid
    for sd in shared_dims:
        for pdim in private_dims:
            # 2) Load model bundle
            run_name = f"{run_prefix}_{subj}_sd{sd}_pd{pdim}_map"
            model_path = os.path.join(model_save_dir, subj, f"{run_name}_bundle.pth")
            if not os.path.isfile(model_path):
                print(f"[WARN] Missing model for {subj}: {model_path}")
                continue

            checkpoint = torch.load(model_path, map_location=device)
            model = checkpoint["model"]
            model.eval()
            del checkpoint  # free memory

            print("Model loaded. Extracting latents...")
            # 3) Extract latents (must be aligned shared)
            with torch.no_grad():
                test_latents = extract_latents_from_test_set_align(model, gpi_test_off, stn_test_off, device)

            shared_gpi = to_np(test_latents["shared_gpi_aligned"])  # (N, T, Ds)
            shared_stn = to_np(test_latents["shared_stn_aligned"])  # (N, T, Ds)
            private_gpi = to_np(test_latents["private_gpi"])        # (N, T, Dp)
            private_stn = to_np(test_latents["private_stn"])        # (N, T, Dp)

            # Basic shape checks
            for name, arr in [("shared_gpi", shared_gpi), ("private_gpi", private_gpi),
                            ("shared_stn", shared_stn), ("private_stn", private_stn)]:
                if arr.ndim != 3:
                    raise ValueError(f"{name} must be (N,T,D). Got {arr.shape}")

            # 4) Build (Z, Y) with lag augmentation per region
            # GPi targets
            Zs_gpi = as_3d(shared_gpi)    # (N, T, Ds)
            Zp_gpi = as_3d(private_gpi)   # (N, T, Dp)
            Zs_stn = as_3d(shared_stn)    # (N, T, Ds)
            Zp_stn = as_3d(private_stn)   # (N, T, Dp)
            

            # Concatenate along FEATURE axis (last dim), not time
            Zsp_gpi_3d = np.concatenate([Zs_gpi, Zp_gpi], axis=2)   # (N, T, Ds+Dp)
            Zsp_stn_3d = np.concatenate([Zs_stn, Zp_stn], axis=2)   # (N, T, Ds+Dp)

            # Flatten (N,T,·) -> (N*T, ·) for decoders
            Zs_gpi_2d  = Zs_gpi.reshape(-1, Zs_gpi.shape[-1])
            Zp_gpi_2d  = Zp_gpi.reshape(-1, Zp_gpi.shape[-1])
            Zsp_gpi_2d = Zsp_gpi_3d.reshape(-1, Zsp_gpi_3d.shape[-1])
            Y_gpi_2d   = Y_gpi.reshape(-1, Y_gpi.shape[-1])

            Zs_stn_2d  = Zs_stn.reshape(-1, Zs_stn.shape[-1])
            Zp_stn_2d  = Zp_stn.reshape(-1, Zp_stn.shape[-1])
            Zsp_stn_2d = Zsp_stn_3d.reshape(-1, Zsp_stn_3d.shape[-1])
            Y_stn_2d   = Y_stn.reshape(-1, Y_stn.shape[-1])

            # ---- CV decode FVE ----
            FVE_S_gpi  = cv_decode_fve(Zs_gpi_2d,  Y_gpi_2d)
            FVE_P_gpi  = cv_decode_fve(Zp_gpi_2d,  Y_gpi_2d)
            FVE_SP_gpi = cv_decode_fve(Zsp_gpi_2d, Y_gpi_2d)

            FVE_S_stn  = cv_decode_fve(Zs_stn_2d,  Y_stn_2d)
            FVE_P_stn  = cv_decode_fve(Zp_stn_2d,  Y_stn_2d)
            FVE_SP_stn = cv_decode_fve(Zsp_stn_2d, Y_stn_2d)

            # 6) Order-free partition (clip tiny negatives)
            US_gpi = safe_clip(FVE_SP_gpi - FVE_P_gpi)
            UP_gpi = safe_clip(FVE_SP_gpi - FVE_S_gpi)
            R_gpi  = safe_clip(FVE_S_gpi + FVE_P_gpi - FVE_SP_gpi)

            US_stn = safe_clip(FVE_SP_stn - FVE_P_stn)
            UP_stn = safe_clip(FVE_SP_stn - FVE_S_stn)
            R_stn  = safe_clip(FVE_S_stn + FVE_P_stn - FVE_SP_stn)

            # 7) Collect rows
            all_rows += [
                dict(subject=subj, side=side, region="GPi",sd=sd, pdim=pdim,
                    FVE_shared=FVE_S_gpi, FVE_private=FVE_P_gpi, FVE_total=FVE_SP_gpi,
                    UniqueShared=US_gpi, UniquePrivate=UP_gpi, Redundant=R_gpi,
                    ),
                dict(subject=subj, side=side, region="STN",sd=sd, pdim=pdim,
                    FVE_shared=FVE_S_stn, FVE_private=FVE_P_stn, FVE_total=FVE_SP_stn,
                    UniqueShared=US_stn, UniquePrivate=UP_stn, Redundant=R_stn,
                    ),
            ]



# -----------------------------
# Save Excel
# -----------------------------
if not all_rows:
    print("No results; check paths/model names.")
else:
    df = pd.DataFrame(all_rows)
    # numeric hygiene
    for c in ["FVE_shared","FVE_private","FVE_total","UniqueShared","UniquePrivate","Redundant"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    out_xlsx = os.path.join(
        excel_save_dir,
        f"{run_prefix}_{side}_variance_partition_sd3-5_pd2-4T.xlsx"
    )
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="all_subjects")

    out_csv = out_xlsx.replace(".xlsx", ".csv")
    df.to_csv(out_csv, index=False)

    print("✅ Excel file saved:", out_xlsx)
    print("📝 CSV file saved  :", out_csv)



#____________Left side
subject_list = ["s508", "s513","s514", "s515","s518","s519","s520","s521","s523"]  # left side
data_save_dir = r"F:\comp_project\Off_tensor_Data_L" #####side
run_prefix="SPIRE_final_L"
side="L"

all_rows = []
for subj in subjects:
    print(f"\nProcessing subject {subj}...")

    # 1) Load observed test tensors (targets)
    off_dir = os.path.join(data_save_dir, subj)
    gpi_test_off = torch.load(os.path.join(off_dir, "gpi_test_off.pt"))   # expected (N, T, Cg)
    stn_test_off = torch.load(os.path.join(off_dir, "stn_test_off.pt"))   # expected (N, T, Cs)
    print(f"Loaded X: GPi {tuple(gpi_test_off.shape)}, STN {tuple(stn_test_off.shape)}")
    Y_gpi  = as_3d(gpi_test_off)  # (N, T, Cg)
    Y_stn  = as_3d(stn_test_off)  # (N, T, Cs)

    # Sweep the (sd, pdim) grid
    for sd in shared_dims:
        for pdim in private_dims:
            # 2) Load model bundle
            run_name = f"{run_prefix}_{subj}_sd{sd}_pd{pdim}_map"
            model_path = os.path.join(model_save_dir, subj, f"{run_name}_bundle.pth")
            if not os.path.isfile(model_path):
                print(f"[WARN] Missing model for {subj}: {model_path}")
                continue

            checkpoint = torch.load(model_path, map_location=device)
            model = checkpoint["model"]
            model.eval()
            del checkpoint  # free memory

            print("Model loaded. Extracting latents...")
            # 3) Extract latents (must be aligned shared)
            with torch.no_grad():
                test_latents = extract_latents_from_test_set_align(model, gpi_test_off, stn_test_off, device)

            shared_gpi = to_np(test_latents["shared_gpi_aligned"])  # (N, T, Ds)
            shared_stn = to_np(test_latents["shared_stn_aligned"])  # (N, T, Ds)
            private_gpi = to_np(test_latents["private_gpi"])        # (N, T, Dp)
            private_stn = to_np(test_latents["private_stn"])        # (N, T, Dp)

            # Basic shape checks
            for name, arr in [("shared_gpi", shared_gpi), ("private_gpi", private_gpi),
                            ("shared_stn", shared_stn), ("private_stn", private_stn)]:
                if arr.ndim != 3:
                    raise ValueError(f"{name} must be (N,T,D). Got {arr.shape}")

            # 4) Build (Z, Y) with lag augmentation per region
            # GPi targets
            Zs_gpi = as_3d(shared_gpi)    # (N, T, Ds)
            Zp_gpi = as_3d(private_gpi)   # (N, T, Dp)
            Zs_stn = as_3d(shared_stn)    # (N, T, Ds)
            Zp_stn = as_3d(private_stn)   # (N, T, Dp)
            

            # Concatenate along FEATURE axis (last dim), not time
            Zsp_gpi_3d = np.concatenate([Zs_gpi, Zp_gpi], axis=2)   # (N, T, Ds+Dp)
            Zsp_stn_3d = np.concatenate([Zs_stn, Zp_stn], axis=2)   # (N, T, Ds+Dp)

            # Flatten (N,T,·) -> (N*T, ·) for decoders
            Zs_gpi_2d  = Zs_gpi.reshape(-1, Zs_gpi.shape[-1])
            Zp_gpi_2d  = Zp_gpi.reshape(-1, Zp_gpi.shape[-1])
            Zsp_gpi_2d = Zsp_gpi_3d.reshape(-1, Zsp_gpi_3d.shape[-1])
            Y_gpi_2d   = Y_gpi.reshape(-1, Y_gpi.shape[-1])

            Zs_stn_2d  = Zs_stn.reshape(-1, Zs_stn.shape[-1])
            Zp_stn_2d  = Zp_stn.reshape(-1, Zp_stn.shape[-1])
            Zsp_stn_2d = Zsp_stn_3d.reshape(-1, Zsp_stn_3d.shape[-1])
            Y_stn_2d   = Y_stn.reshape(-1, Y_stn.shape[-1])

            # ---- CV decode FVE ----
            FVE_S_gpi  = cv_decode_fve(Zs_gpi_2d,  Y_gpi_2d)
            FVE_P_gpi  = cv_decode_fve(Zp_gpi_2d,  Y_gpi_2d)
            FVE_SP_gpi = cv_decode_fve(Zsp_gpi_2d, Y_gpi_2d)

            FVE_S_stn  = cv_decode_fve(Zs_stn_2d,  Y_stn_2d)
            FVE_P_stn  = cv_decode_fve(Zp_stn_2d,  Y_stn_2d)
            FVE_SP_stn = cv_decode_fve(Zsp_stn_2d, Y_stn_2d)

            # 6) Order-free partition (clip tiny negatives)
            US_gpi = safe_clip(FVE_SP_gpi - FVE_P_gpi)
            UP_gpi = safe_clip(FVE_SP_gpi - FVE_S_gpi)
            R_gpi  = safe_clip(FVE_S_gpi + FVE_P_gpi - FVE_SP_gpi)

            US_stn = safe_clip(FVE_SP_stn - FVE_P_stn)
            UP_stn = safe_clip(FVE_SP_stn - FVE_S_stn)
            R_stn  = safe_clip(FVE_S_stn + FVE_P_stn - FVE_SP_stn)

            # 7) Collect rows
            all_rows += [
                dict(subject=subj, side=side, region="GPi",sd=sd, pdim=pdim,
                    FVE_shared=FVE_S_gpi, FVE_private=FVE_P_gpi, FVE_total=FVE_SP_gpi,
                    UniqueShared=US_gpi, UniquePrivate=UP_gpi, Redundant=R_gpi,
                    ),
                dict(subject=subj, side=side, region="STN",sd=sd, pdim=pdim,
                    FVE_shared=FVE_S_stn, FVE_private=FVE_P_stn, FVE_total=FVE_SP_stn,
                    UniqueShared=US_stn, UniquePrivate=UP_stn, Redundant=R_stn,
                    ),
            ]



# -----------------------------
# Save Excel
# -----------------------------
if not all_rows:
    print("No results; check paths/model names.")
else:
    df = pd.DataFrame(all_rows)
    # numeric hygiene
    for c in ["FVE_shared","FVE_private","FVE_total","UniqueShared","UniquePrivate","Redundant"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    out_xlsx = os.path.join(
        excel_save_dir,
        f"{run_prefix}_{side}_variance_partition_sd3-5_pd2-4T.xlsx"
    )
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="all_subjects")

    out_csv = out_xlsx.replace(".xlsx", ".csv")
    df.to_csv(out_csv, index=False)

    print("✅ Excel file saved:", out_xlsx)
    print("📝 CSV file saved  :", out_csv)