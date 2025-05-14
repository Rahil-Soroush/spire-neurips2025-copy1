import os 
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import umap

save_test_latent_dir = r"F:\comp_project\2region_models\test_latents_onstim_s3p5"
subj = "s513"
setting = "GPi1_L12"
subj_path = os.path.join(save_test_latent_dir, subj)
path_latents = os.path.join(subj_path, setting)
        
# Extract the side (first character after the underscore)
try:
    side = setting.split("_")[1][0]  # 'L' from 'L12'
except IndexError:
    side = "?"  # fallback if the format is unexpected

print(f"Subject: {subj} | Setting: {setting} | Side: {side}")

#load latents
label_map = {0: "Off", 1: "85Hz", 2: "185Hz", 3: "250Hz"}
GPi_shared_gpi = torch.load(os.path.join(path_latents, "shared_gpi.pt"))
GPi_shared_stn = torch.load(os.path.join(path_latents, "shared_stn.pt"))
GPi_private_gpi = torch.load(os.path.join(path_latents, "private_gpi.pt"))
GPi_private_stn = torch.load(os.path.join(path_latents, "private_stn.pt"))

print(f"GPi Test latents Loaded")

setting = "VoSTN_L12"
subj_path = os.path.join(save_test_latent_dir, subj)
path_latents = os.path.join(subj_path, setting)
        
# Extract the side (first character after the underscore)
try:
    side = setting.split("_")[1][0]  # 'L' from 'L12'
except IndexError:
    side = "?"  # fallback if the format is unexpected

print(f"Subject: {subj} | Setting: {setting} | Side: {side}")

#load latents
label_map = {0: "Off", 1: "85Hz", 2: "185Hz", 3: "250Hz"}
STN_shared_gpi = torch.load(os.path.join(path_latents, "shared_gpi.pt"))
STN_shared_stn = torch.load(os.path.join(path_latents, "shared_stn.pt"))
STN_private_gpi = torch.load(os.path.join(path_latents, "private_gpi.pt"))
STN_private_stn = torch.load(os.path.join(path_latents, "private_stn.pt"))

#displacemnet due to stimulation
label_keys = ["Off", "185Hz"]
GPi_mean_latents = {
    'Shared GPi': {
        k: GPi_shared_gpi[k].mean(dim=0).cpu().numpy() for k in label_keys if k in GPi_shared_gpi and len(GPi_shared_gpi[k]) > 0
    },
    'Private GPi': {
        k: GPi_private_gpi[k].mean(dim=0).cpu().numpy() for k in label_keys if k in GPi_private_gpi and len(GPi_private_gpi[k]) > 0
    },
    'Shared STN': {
        k: GPi_shared_stn[k].mean(dim=0).cpu().numpy() for k in label_keys if k in GPi_shared_stn and len(GPi_shared_stn[k]) > 0
    },
    'Private STN': {
        k: GPi_private_stn[k].mean(dim=0).cpu().numpy() for k in label_keys if k in GPi_private_stn and len(GPi_private_stn[k]) > 0
    },
}

STN_mean_latents = {
    'Shared GPi': {
        k: STN_shared_gpi[k].mean(dim=0).cpu().numpy() for k in label_keys if k in STN_shared_gpi and len(STN_shared_gpi[k]) > 0
    },
    'Private GPi': {
        k: STN_private_gpi[k].mean(dim=0).cpu().numpy() for k in label_keys if k in STN_private_gpi and len(STN_private_gpi[k]) > 0
    },
    'Shared STN': {
        k: STN_shared_stn[k].mean(dim=0).cpu().numpy() for k in label_keys if k in STN_shared_stn and len(STN_shared_stn[k]) > 0
    },
    'Private STN': {
        k: STN_private_stn[k].mean(dim=0).cpu().numpy() for k in label_keys if k in STN_private_stn and len(STN_private_stn[k]) > 0
    },
}

mean_latents = {}
latent_types = ['Shared GPi', 'Private GPi', 'Shared STN', 'Private STN']
for lt in latent_types:
    mean_latents[lt] = {
        'Off': GPi_mean_latents[lt]['Off'],      # Use GPi Off for both comparisons
        'GPi': GPi_mean_latents[lt]['185Hz'],    # GPi 185Hz stimulation
        'STN': STN_mean_latents[lt]['185Hz'],    # STN 185Hz stimulation
    }
# Config
latent_types = ['Shared GPi', 'Private GPi', 'Shared STN', 'Private STN']
color_map = {
    'GPi': 'firebrick',
    'STN': 'midnightblue'
}

# Create Plotly Subplot
fig = make_subplots(
    rows=2, cols=2,
    specs=[[{'type': 'scatter3d'}]*2]*2,
    subplot_titles=latent_types
)

for idx, lt in enumerate(latent_types):
    data = np.vstack([
        mean_latents[lt]['Off'],
        mean_latents[lt]['GPi'],
        mean_latents[lt]['STN']
    ])

    reducer = umap.UMAP(n_components=3, n_neighbors=30, min_dist=0.05, random_state=42)
    emb = reducer.fit_transform(data)
    T = mean_latents[lt]['Off'].shape[0]

    emb_off, emb_gpi, emb_stn = emb[:T], emb[T:2*T], emb[2*T:3*T]
    delta_gpi = emb_gpi - emb_off
    delta_stn = emb_stn - emb_off

    row, col = idx // 2 + 1, idx % 2 + 1

    for label, delta, style in zip(["GPi", "STN"], [delta_gpi, delta_stn], ['solid', 'dash']):
        fig.add_trace(
            go.Scatter3d(
                x=delta[:, 0], y=delta[:, 1], z=delta[:, 2],
                mode="lines",
                line=dict(color=color_map[label], width=4, dash=style),
                name=label if row == 1 and col == 1 else None,
                showlegend=(row == 1 and col == 1)
            ),
            row=row, col=col
        )

# Layout
fig.update_layout(
    height=900,
    width=1100,
    title_text="3D UMAP – Latent Shift from Off to GPi vs STN Stimulation",
    margin=dict(l=10, r=10, b=10, t=60),
    showlegend=True,
    scene=dict(
        xaxis_title='UMAP 1',
        yaxis_title='UMAP 2',
        zaxis_title='UMAP 3',
    )
)

fig.show()