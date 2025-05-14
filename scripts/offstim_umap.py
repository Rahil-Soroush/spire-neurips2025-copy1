import os
import torch
from sklearn.decomposition import PCA
from src.visualization.umap_utils import (
    extract_latents_from_test_set,
    flatten_latents,
    subsample_group,
    run_umap_and_label,
)
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

data_save_dir = r"F:\comp_project\Off_tensor_Data_R" #####side
model_save_dir = r"F:\comp_project\2region_models"

shared_dim = 3
subj = "s508"
# 1. Load the test tensors
Off_test_data_dir = os.path.join(data_save_dir, subj)
gpi_test_off = torch.load(os.path.join(Off_test_data_dir, "gpi_test_off.pt"))
stn_test_off = torch.load(os.path.join(Off_test_data_dir, "stn_test_off.pt"))

print(f"Loaded: {gpi_test_off.shape}, {stn_test_off.shape}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_save_path = os.path.join(model_save_dir, subj, f"model_P20_sD3_pD5_R.pt") ########side
checkpoint = torch.load(model_save_path)

model = checkpoint['model']
model.eval()

print(f"model Loaded")

shared_gpi, shared_stn, private_gpi,private_stn = extract_latents_from_test_set(model, gpi_test_off, stn_test_off, device)

shared_gpi_f = flatten_latents(shared_gpi).numpy()  # (N*T, D_shared)
shared_stn_f = flatten_latents(shared_stn).numpy()  # (N*T, D_shared)
private_gpi_f = flatten_latents(private_gpi).numpy() # (N*T, D_private)
private_stn_f = flatten_latents(private_stn).numpy() # (N*T, D_private)

# Step 2: Project each to 3D using PCA
proj_dim = 3
pca = PCA(n_components=proj_dim)

shared_gpi_proj = pca.fit_transform(shared_gpi_f)
private_gpi_proj = pca.fit_transform(private_gpi_f)
shared_stn_proj = pca.fit_transform(shared_stn_f)
private_stn_proj = pca.fit_transform(private_stn_f)

# Subsample each group
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
font = dict(family="Arial", size=18)
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

# ---- Style layout ----
fig.update_layout(
    height=550, width=1600,
    margin=dict(l=0, r=0, b=0, t=60),
    # title=dict(
    #     text="3D UMAP of Shared and Private Latents",
    #     font=font, x=0.5
    # ),
    legend=dict(
        font=font,
        orientation="h",
        y=1.12,
        x=0.5,
        xanchor="center",
        itemsizing='constant',  # ensures consistent scaling
        bordercolor="LightGray",
        borderwidth=1
    )
)

# Apply camera and hide tick labels
for i in range(1, 4):
    fig[f"layout"][f"scene{i}"].update(
        xaxis_title='UMAP1', yaxis_title='UMAP2', zaxis_title='UMAP3',
        xaxis=dict(title_font=font, showticklabels=False),
        yaxis=dict(title_font=font, showticklabels=False),
        zaxis=dict(title_font=font, showticklabels=False),
        camera=camera_settings
    )

# ---- Save PNG and PDF ----
fig.write_image(r"F:\comp_project\2region_models\figures\s508_R_offstim_2000samp_umap_latents_combined.png", scale=4)
fig.write_image(r"F:\comp_project\2region_models\figures\s508_R_offstim_2000samp_umap_latents_combined.pdf", scale=4)

fig.show()