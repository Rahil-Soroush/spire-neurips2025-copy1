#after deciding the best dim for each subject, we choose two representative subjects to visualize their UMAP latents and time domain latents
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]   # repo root
sys.path.insert(0, str(root / "src"))

import os
import torch
from sklearn.decomposition import PCA
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from models.spire_model import SPIREAutoencoder, LatentEncoder, LatentDecoder, ConvAlign1D
from evaluate_real import extract_latents_from_test_set_align, to_np, flatten_latents
from visualization.real_visualizer import subsample_group, run_umap_and_label, camera_from_angles, plot_latent_traces

data_save_dir = r"F:\comp_project\Off_tensor_Data_R" #####side
model_save_dir = r"F:\comp_project\2region_models_SPIRE_dimSweep"
base = r"F:\comp_project\2region_models_SPIRE_dimSweep\figures"
os.makedirs(base, exist_ok=True)

subj = "s514" # repeat for 520 with these dimesnions: 3,4
sd = 3
pdim = 3
side = "R"
run_prefix="SPIRE_final_R"

# 1. Load the test tensors
Off_test_data_dir = os.path.join(data_save_dir, subj)
gpi_test_off = torch.load(os.path.join(Off_test_data_dir, "gpi_test_off.pt"))
stn_test_off = torch.load(os.path.join(Off_test_data_dir, "stn_test_off.pt"))
print(f"Loaded: {gpi_test_off.shape}, {stn_test_off.shape}")

#2. Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run_name = f"{run_prefix}_{subj}_sd{sd}_pd{pdim}_map"
model_save_path = os.path.join(model_save_dir,subj, f"{run_name}_bundle.pth")
checkpoint = torch.load(model_save_path)
model = checkpoint['model']
model.eval()
print(f"model Loaded")

#3. extract latents
test_latents = extract_latents_from_test_set_align(model, gpi_test_off, stn_test_off, device)
shared_gpi = test_latents['shared_gpi_aligned']
shared_stn = test_latents['shared_stn_aligned']
private_gpi = test_latents['private_gpi']
private_stn = test_latents['private_stn']

#############_____UMAP plot
shared_gpi_f = to_np(flatten_latents(shared_gpi))
shared_stn_f = to_np(flatten_latents(shared_stn))
private_gpi_f = to_np(flatten_latents(private_gpi))
private_stn_f = to_np(flatten_latents(private_stn))

# (optional) separate PCA instances (not required, but cleaner)
shared_gpi_proj  = PCA(n_components=3).fit_transform(shared_gpi_f)
private_gpi_proj = PCA(n_components=3).fit_transform(private_gpi_f)
shared_stn_proj  = PCA(n_components=3).fit_transform(shared_stn_f)
private_stn_proj = PCA(n_components=3).fit_transform(private_stn_f)

# Subsample each group with 2000 samples
shared_gpi_sub, lbl1 = subsample_group(shared_gpi_proj, "Shared GPi")
private_gpi_sub, lbl2 = subsample_group(private_gpi_proj, "Private GPi")
shared_stn_sub, lbl3 = subsample_group(shared_stn_proj, "Shared STN")
private_stn_sub, lbl4 = subsample_group(private_stn_proj, "Private STN")

color_map = {
    "Private GPi": "forestgreen",
    "Private STN": "darkorange",
    "Shared GPi": "steelblue",
    "Shared STN": "deeppink"
}

# ---- Run UMAP on the three groups ----
X_shared = np.concatenate([shared_gpi_sub, shared_stn_sub], axis=0)
labels_shared = lbl1 + lbl3
df_shared = run_umap_and_label(X_shared, labels_shared)

X_gpi = np.concatenate([shared_gpi_sub, private_gpi_sub], axis=0)
labels_gpi = lbl1 + lbl2
df_gpi = run_umap_and_label(X_gpi, labels_gpi)

X_stn = np.concatenate([shared_stn_sub, private_stn_sub], axis=0)
labels_stn = lbl3 + lbl4
df_stn = run_umap_and_label(X_stn, labels_stn)

# ---- Create the 3-panel 3D UMAP plot ----
fig = make_subplots(
    rows=1, cols=3, specs=[[{'type': 'scatter3d'}]*3],
    subplot_titles=[
        "Shared GPi & STN",
        "GPi Latents",
        "STN Latents"
    ]
)

# Custom fonts & camera
subplot_title_font = dict(family="Times New Roman", size=24)
camera_settings = dict(eye=dict(x=1.2, y=1.2, z=0.6))

# Track which labels are already added to legend
legend_labels = set()

# ---- Add traces to each subplot ----
for df, col in zip([df_shared, df_gpi, df_stn], [1, 2, 3]):
    for label in df["label"].unique():
        data = df[df["label"] == label]
        show_legend = label not in legend_labels
        fig.add_trace(go.Scatter3d(
            x=data["UMAP1"], y=data["UMAP2"], z=data["UMAP3"],
            mode='markers',
            marker=dict(size=2.5, color=color_map[label]),
            name=label,
            showlegend=show_legend
        ), row=1, col=col)
        legend_labels.add(label)

fig.update_layout(height=550,width=1000,margin=dict(l=10, r=10, b=10, t=65),legend=dict(font=subplot_title_font, orientation="h",y=1.17, x=0.5, xanchor="center",itemsizing='constant',  # ensures consistent scaling
        bordercolor="LightGray",
        borderwidth=1))
fig.update_annotations(font=subplot_title_font)

axis_font = dict(family="Times New Roman", size=20)
for scene_key in ["scene", "scene2", "scene3"]:
    fig.layout[scene_key].update(
        xaxis_title='UMAP1', yaxis_title='UMAP2', zaxis_title='UMAP3',
        xaxis=dict(title_font=axis_font, showticklabels=False, showgrid=True, gridcolor="rgba(0,0,0,0.3)", showbackground=False),
        yaxis=dict(title_font=axis_font, showticklabels=False, showgrid=True, gridcolor="rgba(0,0,0,0.3)", showbackground=False),
        zaxis=dict(title_font=axis_font, showticklabels=False, showgrid=True, gridcolor="rgba(0,0,0,0.3)", showbackground=False)
    )

# this setting meeds to be manually adjusted for each subject
cam_STN = camera_from_angles(az_deg=-45, elev_deg=22, r=2.7) #-45, 22
cam = camera_from_angles(az_deg=130, elev_deg=70, r=2.7)
fig.update_layout(
    scene = dict(camera=cam),
    scene2= dict(camera=cam),
    scene3= dict(camera=cam_STN),
)

fig.show()

# ---- Save PNG and PDF ----

fname = f"{run_prefix}_{subj}_{side}_sd{sd}_pd{pdim}_offstim_2000samp_umap_latents_combined.png"
fig.write_image(rf"{base}\{fname}",height=550,width=1000,scale=1)  

fname = f"{run_prefix}_{subj}_{side}_sd{sd}_pd{pdim}_offstim_2000samp_umap_latents_combined.pdf"
fig.write_image(rf"{base}\{fname}",height=550,width=1000,scale=1)  



################# --------- TIME domain plots
trial = 10 
time_fig, _ = plot_latent_traces(
    shared_gpi, shared_stn, private_gpi, private_stn,
    trial=trial, dims=(0,1,2), tlim=None,
    width_in=6.9,            # two-column width in inches
    row_height_in=1.4,       # per-row height
    base_fontsize=9,
    legend_headroom=0.01,    # space for top legends (per column)
    line_width=1.2,
    savepath=None, dpi=300
)


fname = f"{subj}_{side}_sd{sd}_pd{pdim}_trial{trial}_latent_time_traces.pdf"
time_fig.savefig(rf"{base}\{fname}", dpi=300, bbox_inches="tight", format="pdf")  