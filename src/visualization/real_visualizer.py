import numpy as np
import pandas as pd
import umap
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def subsample_group(X, label, target_n=2000, seed=42):
    np.random.seed(seed)
    idx = np.random.choice(len(X), size=min(target_n, len(X)), replace=False)
    return X[idx], [label] * len(idx)

def prepare_umap_df(X, labels):
    return pd.DataFrame(X, columns=['UMAP1', 'UMAP2', 'UMAP3']).assign(label=labels)

# --- UMAP runner with neighbors guard (prevents "n_neighbors > n_samples")
def run_umap_and_label(X, labels, n_neighbors=20):
    n_neighbors = max(2, min(n_neighbors, len(X) - 1))  # keep valid
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1,
                        n_components=3, random_state=42)
    X_umap = reducer.fit_transform(X)
    df = pd.DataFrame(X_umap, columns=["UMAP1", "UMAP2", "UMAP3"])
    df["label"] = labels
    return df

def camera_from_angles(az_deg=45, elev_deg=20, r=1.8,
                       up=(0,0,1), center=(0,0,0)):
    """
    az_deg  : yaw/azimuth around +z (0° = +x axis; 90° = +y)
    elev_deg: elevation above the xy-plane (0° = horizon; 90° = top-down)
    r       : distance of the camera from the origin
    """
    az, el = math.radians(az_deg), math.radians(elev_deg)
    x = r * math.cos(el) * math.cos(az)
    y = r * math.cos(el) * math.sin(az)
    z = r * math.sin(el)
    return dict(eye=dict(x=x, y=y, z=z),
                up=dict(x=up[0], y=up[1], z=up[2]),
                center=dict(x=center[0], y=center[1], z=center[2]))

def _zscore2(x, axis=0, eps=1e-8):
    m = x.mean(axis=axis, keepdims=True); s = x.std(axis=axis, keepdims=True) + eps
    return (x - m) / s

def plot_latent_traces(
    shared_gpi, shared_stn, private_gpi, private_stn,
    trial=None, dims=(0,1,2), tlim=None,
    width_in=6.9,            # two-column width in inches
    row_height_in=0.9,       # per-row height
    base_fontsize=9,
    legend_headroom=0.11,    # space for top legends (per column)
    line_width=1.2,
    savepath=None, dpi=300
):
    """
    Inputs:
      shared_* : (N, T, Ds)  zscored per-dim in time before plotting
      private_*: (N, T, Dp)
    """
    # Colors (keep your palette)
    c_shared_gpi = "steelblue"
    c_shared_stn = "deeppink"
    c_priv_gpi   = "forestgreen"
    c_priv_stn   = "darkorange"

    N, T, Ds = shared_gpi.shape
    Dp = private_gpi.shape[2]
    if trial is None:
        # pick the trial with max energy in shared_gpi
        energy = (shared_gpi**2).sum(axis=(1,2))
        trial = int(np.argmax(energy))
    t0, t1 = (0, T) if tlim is None else tlim

    # Z-score across time, per latent dim
    def z(a): return _zscore2(a[trial, t0:t1], axis=0)

    sg = z(shared_gpi)  # (T,Ds)
    ss = z(shared_stn)
    pg = z(private_gpi)
    ps = z(private_stn)

    nrows = len(dims)
    height_in = max(2.6, nrows * row_height_in)
    rc = {
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "text.usetex": False,
        "mathtext.fontset": "stix",              # Times-like math
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L", "TeX Gyre Termes", "DejaVu Serif"],
        "font.size": base_fontsize,
        "axes.labelsize": base_fontsize,
        "axes.titlesize": base_fontsize,
        "xtick.labelsize": max(base_fontsize-1, 7),
        "ytick.labelsize": max(base_fontsize-1, 7),
        "legend.fontsize": max(base_fontsize-1, 7),
        "lines.linewidth": line_width,
    }
    with mpl.rc_context(rc):
        # two columns: left=shared, right=private
        fig, axes = plt.subplots(nrows=nrows, ncols=2,
                                 figsize=(width_in, height_in),
                                 sharex=True)
        if nrows == 1:
            axes = np.array([axes])

         # Title and legend font properties (explicit Times)
        title_prop  = FontProperties(family="Times New Roman", size=base_fontsize)
        legend_prop = FontProperties(family="Times New Roman", size=max(base_fontsize-1, 7))
        ylabel_prop = FontProperties(family="Times New Roman", size=base_fontsize)

        # Collect handles for column-wise legends
        handles_left, labels_left = None, None
        handles_right, labels_right = None, None

        # Plot each dim as a row
        for r, d in enumerate(dims):
            # --- Shared (left column) ---
            axL = axes[r, 0]
            h1 = axL.plot(sg[:, d], color=c_shared_gpi, lw=line_width, label="GPi (shared)")[0]
            h2 = axL.plot(ss[:, d], color=c_shared_stn, lw=line_width, label="STN (shared)")[0]
            axL.set_title(f"Shared dim {d} (GPi vs STN)", pad=2, fontproperties=title_prop)
            axL.grid(alpha=0.3)
            if handles_left is None:
                handles_left = [h1, h2]
                labels_left = [h.get_label() for h in handles_left]

            # --- Private (right column) ---
            axR = axes[r, 1]
            if d < Dp:
                h3 = axR.plot(pg[:, d], color=c_priv_gpi, lw=line_width, label="GPi (private)")[0]
                h4 = axR.plot(ps[:, d], color=c_priv_stn, lw=line_width, label="STN (private)")[0]
            axR.set_title(f"Private dim {d} (GPi vs STN)", pad=2, fontproperties=title_prop)
            axR.grid(alpha=0.3)
            if handles_right is None:
                handles_right = [h3, h4]
                labels_right = [h.get_label() for h in handles_right]

        # # Axis labels
        # for r in range(nrows):
        #     axes[r, 0].set_ylabel("z-scored latent (a.u.)")

        # 1) Remove per-axis ylabels
        for r in range(nrows):
            for c in range(2):
                axes[r, c].set_ylabel(None)

        # 2) Reserve a bit more left margin AND the top band for legends
        top_rect = 1.0 - legend_headroom
        plt.tight_layout(rect=[0.03, 0.0, 1.0, top_rect])  # left=6%

        # 3) Single y-label on the left, vertically centered
        fig.text(0.002, 0.5, "z-scored latent (a.u.)",
                rotation="vertical", va="center", ha="left",
                fontproperties=ylabel_prop)

        axes[-1, 0].set_xlabel("Time (samples)")
        axes[-1, 1].set_xlabel("Time (samples)")

        # Share y within each column for easy comparison
        # (match limits to the tightest across rows per column)
        for c in [0, 1]:
            ymin = min(ax.get_ylim()[0] for ax in axes[:, c])
            ymax = max(ax.get_ylim()[1] for ax in axes[:, c])
            for r in range(nrows):
                axes[r, c].set_ylim(ymin, ymax)

        # Left legend (shared)
        fig.legend(handles_left, labels_left, ncol=2, loc="center",
                   bbox_to_anchor=(0.25, top_rect + legend_headroom/2), frameon=False, prop=legend_prop)
        # Right legend (private)
        fig.legend(handles_right, labels_right, ncol=2, loc="center",
                   bbox_to_anchor=(0.75, top_rect + legend_headroom/2), frameon=False, prop=legend_prop)

        if savepath:
            fig.savefig(savepath, bbox_inches="tight", dpi=dpi)

    return fig, axes