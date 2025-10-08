import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]   # repo root
sys.path.insert(0, str(root / "src"))

import os
import torch
import scipy
import numpy as np
import matplotlib.pyplot as plt
from data.synth_data import load_saved_dataset
from data.data_loader import build_dataset_with_lag
from evaluate_synth import extract_latents_from_test_set
from visualization.synth_visualizer import plot_shared1_mean_sem
from models.spire_model import SPIREAutoencoder, LatentEncoder, LatentDecoder, ConvAlign1D

#-------------load data--------
data_save_dir = r"F:\comp_project\synthecticData\dataT"#your target directory to save datasets
DATA_plotting = {

    #D1 adds region-mismatched nonlinear warps and bilinear mixing with 1/f noise and AR(1) latents
    "D1_warp_nonLinear": dict(
        n_trials=100, T=250, fs=500, shared_dim=3, private_dim=3,
        bipolarize=True,
        spatial_mixing="random",              # or "gaussian" if you want harsher bipolar effects
        nonlin_mode="gain+bilinear", gain_g=0.8, bilinear_beta=0.8,
        interaction_strength=0.2,
        shared_variant="identity",
        shared_warp="region_mismatch",
        add_one_over_f=True, one_over_f_strength=0.2,
        ar_stage="latents", ar1_rho=0.3,
        sensor_noise=0.02,
        add_row_common=False, add_common_mode=False,
        seed=702,
    ),
    
}

ALL_DATASETS = []
for regime_name, config in DATA_plotting.items():
    ds = load_saved_dataset(regime_name, config, data_save_dir)
    ALL_DATASETS.append(ds)

dataset =ALL_DATASETS[0]
regime = dataset["regime"]
data = dataset["data"]

#-------------load SPIRE model--------
model_save_dir = r"F:\comp_project\synthecticData\model_val_sweep3"
model_save_path = os.path.join(model_save_dir, f"S1_warp_gainbilin_SPIRE_synth_E500_seed702_bundle.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(model_save_path)
model = checkpoint['model']
model.eval()

print(f"SPIRE model Loaded")


#-------------load DLAG model (trained using MATLAB and their published demo code)--------
DLAG_fitted_path = r"F:\comp_project\synthecticData\DLAG_fitted"
mat = scipy.io.loadmat(os.path.join(DLAG_fitted_path, 'S1_warp_gainbilin_500Iter_jitter.mat'))
latents = mat['dlag_latents_struct']

# Each field can be accessed (using numpy structured array notation)
DLAG_g1_shared  = latents['g1_shared'][0,0]    # shape: (nTrials, 4, T)
DLAG_g1_private = latents['g1_private'][0,0]   # shape: (nTrials, 2, T)
DLAG_g2_shared  = latents['g2_shared'][0,0]    # shape: (nTrials, 4, T)
DLAG_g2_private = latents['g2_private'][0,0]   # shape: (nTrials, 2, T)

# Permute axes to (nTrials, T, dim)
DLAG_g1_shared  = np.transpose(DLAG_g1_shared, (0, 2, 1))
DLAG_g1_private = np.transpose(DLAG_g1_private, (0, 2, 1))
DLAG_g2_shared  = np.transpose(DLAG_g2_shared, (0, 2, 1))
DLAG_g2_private = np.transpose(DLAG_g2_private, (0, 2, 1))

#-------------extracting latents and plotting--------
figure_save_dir = r"F:\comp_project\synthecticData\figuresT"

lags = 3
#ground truth latents (n_trials, D, T_gt)
gt_shared1_all = data['gt_shared1']
gt_shared2_all = data['gt_shared2']
gt_private1_all = data['gt_private1']
gt_private2_all = data['gt_private2']
gt_latents = {
    "shared1": gt_shared1_all,    # (N, D1, Tg)
    "shared2": gt_shared2_all,    # (N, D2, Tg)
    "private1": gt_private1_all,  # (N, D3, Tg)
    "private2": gt_private2_all,  # (N, D4, Tg)
}

#SPIRE latents (n_trials, T_model, D)
# -------- build inputs exactly like training --------
reg1_R, reg2_R = build_dataset_with_lag(data["region1"], data["region2"], lags=lags)
reg1_tensor = torch.tensor(reg1_R, dtype=torch.float32).permute(0, 2, 1).to(device)  # (N,T,C)
reg2_tensor = torch.tensor(reg2_R, dtype=torch.float32).permute(0, 2, 1).to(device)
model = model.to(device).eval()
shared_reg1, shared_reg2, private_reg1,private_reg2 = extract_latents_from_test_set(model, reg1_tensor, reg2_tensor, device) 

fig, axes, sem_corrs = plot_shared1_mean_sem(
    shared_reg1, DLAG_g1_shared, gt_shared1_all,
    dims=(0,1,2),
    savepath=os.path.join(figure_save_dir, "D1_shared1_mean_semT.pdf"),
    width_in=4, base_fontsize=5,row_height_in=0.1,line_width=0.9, panel_labels=False, legend_headroom=0.001
)

# # Save as PDF
# figure_save_dir = r"F:\comp_project\synthecticData\figures"
# os.path.join(figure_save_dir, "D1_shared1_mean_sem.pdf")
# plt.savefig(os.path.join(figure_save_dir, "D1_shared1_mean_sem.pdf"), format="pdf", bbox_inches="tight")


plt.show()