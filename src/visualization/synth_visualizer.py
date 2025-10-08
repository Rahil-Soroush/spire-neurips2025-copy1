import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
import matplotlib as mpl

# ---------- small utils ----------
def _z(x, axis=None, eps=1e-8):
    x = np.asarray(x)
    if axis is None:
        m = x.mean()
        s = x.std() + eps
    else:
        m = x.mean(axis=axis, keepdims=True)
        s = x.std(axis=axis, keepdims=True) + eps
    return (x - m) / s

def _maybe_item(x):
    try:
        # unwrap 0-d object arrays from npz/mat loads
        if isinstance(x, np.ndarray) and x.dtype == object and x.ndim == 0:
            return x.item()
    except Exception:
        pass
    return x

def _unwrap(ds):
    """Return raw generator dict (keys: 'region1', 'gt_shared1', ...)."""
    if isinstance(ds, dict) and "data" in ds:
        raw = _maybe_item(ds["data"])
        return raw if isinstance(raw, dict) else ds
    return ds

def _get_config(ds):
    """Prefer wrapper config; else inner; unwrap if needed; always return dict."""
    cand = None
    if isinstance(ds, dict) and "config" in ds:
        cand = _maybe_item(ds["config"])
    if not isinstance(cand, dict) and isinstance(ds, dict) and "data" in ds:
        inner = _maybe_item(ds["data"])
        if isinstance(inner, dict) and "config" in inner:
            cand = _maybe_item(inner["config"])
    return cand if isinstance(cand, dict) else {}

def windowed_best_lag(x, y, max_lag=12, win=60, step=10):
    """
    Compute best lag over time via windowed cross-correlation.
    x,y: (T,) arrays (z-scored already is best)
    Returns times, best_lags
    """
    T = len(x)
    centers, lags = [], []
    for start in range(0, T - win + 1, step):
        end = start + win
        xx = x[start:end]
        yy = y[start:end]
        # test lags in [-max_lag..max_lag]
        lag_vals = []
        for L in range(-max_lag, max_lag + 1):
            if L < 0:
                v = np.corrcoef(xx[-L:], yy[:win+L])[0,1]
            elif L > 0:
                v = np.corrcoef(xx[:win-L], yy[L:])[0,1]
            else:
                v = np.corrcoef(xx, yy)[0,1]
            lag_vals.append(v)
        best = np.argmax(np.nan_to_num(lag_vals, nan=-np.inf))
        centers.append(start + win//2)
        lags.append(best - max_lag)
    return np.array(centers), np.array(lags)

def render_dataset_row(fig, outer_gs_cell, ds, trial_idx=0, secs=2.0,
                       latent_ylim=(-2.5, 2.0)):
    """
    Row layout:
      Col 1 (left):  ONE plot — observed ch=0 for Region1 (solid) + Region2 (dashed)
      Col 2 (middle): TWO stacked — shared dim 0 (top), private dim 0 (bottom)
      Col 3 (right):  ONE plot — diag: lag-vs-time (if timevary_delay) OR nonlinear relation
    """
    synthetic_colors = {
        "Region1": "mediumpurple",
        "Region2": "saddlebrown",
    }
    ds_raw = _unwrap(ds)
    cfg = _get_config(ds)
    fs = cfg.get("fs", 500)

    # data
    X1 = ds_raw["region1"][trial_idx]
    X2 = ds_raw["region2"][trial_idx]
    sh1 = ds_raw["gt_shared1"][trial_idx]   # (Ds, T)
    sh2 = ds_raw["gt_shared2"][trial_idx]
    pr1 = ds_raw["gt_private1"][trial_idx]  # (Dp, T)
    pr2 = ds_raw["gt_private2"][trial_idx]

    # # ---- crop here ---- to zoom in
    # frac = 3/5
    # T_full = X1.shape[0]
    # T_cut = int(T_full * frac)
    # X1, X2 = X1[:T_cut], X2[:T_cut]
    # sh1, sh2 = sh1[:, :T_cut], sh2[:, :T_cut]
    # pr1, pr2 = pr1[:, :T_cut], pr2[:, :T_cut]
    # # -------------------

    # ---- crop to middle 3/5 of time ----
    T_full = X1.shape[0]
    start = T_full // 5
    end   = 4 * T_full // 5
    X1, X2 = X1[start:end], X2[start:end]
    sh1, sh2 = sh1[:, start:end], sh2[:, start:end]
    pr1, pr2 = pr1[:, start:end], pr2[:, start:end]
    # ------------------------------------

    # window
    T_avail = min(X1.shape[0], X2.shape[0], sh1.shape[1], sh2.shape[1], pr1.shape[1], pr2.shape[1])
    Tseg = min(int(secs * fs), T_avail)
    t = np.arange(Tseg) / fs

    # fixed choices
    ch1 = ch2 = 0
    i_sh1 = i_sh2 = 0
    i_pr1 = i_pr2 = 0

    # ====== layout: 1x3 ======
    gs_row    = outer_gs_cell.subgridspec(1, 3, wspace=0.45, width_ratios=[1.25, 1.00, 1.00])

    # Col 1: observed overlay
    ax_obs = fig.add_subplot(gs_row[0, 0])

    # Col 2: two stacked (shared, private)
    gs_mid  = gs_row[0, 1].subgridspec(2, 1, hspace=1)
    ax_sh   = fig.add_subplot(gs_mid[0, 0])
    ax_pr   = fig.add_subplot(gs_mid[1, 0])

    # Col 3: diagnostic
    ax_diag = fig.add_subplot(gs_row[0, 2])

    # -------- Col 1: observed channel 0 both regions --------
    ax_obs.plot(t, _z(X1[:Tseg, ch1]), lw=1.2, color=synthetic_colors["Region1"], label="Region 1", alpha=0.95)
    ax_obs.plot(t, _z(X2[:Tseg, ch2]), lw=1.2, color=synthetic_colors["Region2"], ls="--", label="Region 2")
    # ax_obs.set_title("Observed channel 0", fontsize=9)
    ax_obs.set_xlim(0, t[-1]); ax_obs.set_yticks([]); ax_obs.set_xlabel("Time (s)")
    ax_obs.legend(fontsize=7, frameon=False, loc="lower right")

    # -------- Col 2-top: shared dim 0 both regions --------
    ax_sh.plot(t, _z(sh1[i_sh1, :Tseg]), lw=1.1,color=synthetic_colors["Region1"])#, label="Region 1"
    ax_sh.plot(t, _z(sh2[i_sh2, :Tseg]), lw=1.1,color=synthetic_colors["Region2"], ls="--")#, label="Region 2"
    ax_sh.set_title("Shared latent dim 0", fontsize=9)
    ax_sh.set_xlim(0, t[-1]); ax_sh.set_ylim(*latent_ylim)
    ax_sh.set_xlabel("")  # only bottom gets x-label
    # ax_sh.legend(fontsize=7, frameon=False, loc="upper left")

    # -------- Col 2-bottom: private dim 0 both regions --------
    ax_pr.plot(t, _z(pr1[i_pr1, :Tseg]), lw=1.1,color=synthetic_colors["Region1"])#, label="Region 1"
    ax_pr.plot(t, _z(pr2[i_pr2, :Tseg]), lw=1.1,color=synthetic_colors["Region2"], ls="--")#, label="Region 2"
    ax_pr.set_title("Private latent dim 0", fontsize=9)
    ax_pr.set_xlim(0, t[-1]); ax_pr.set_ylim(*latent_ylim)
    ax_pr.set_xlabel("Time (s)")
    # ax_pr.legend(fontsize=7, frameon=False, loc="upper left")

    # -------- Col 3: diagnostic (no heatmap) --------
    # choose best-matching shared dims via correlation to keep it meaningful
    Xz = _z(sh1[:, :Tseg], axis=1)
    Yz = _z(sh2[:, :Tseg], axis=1)
    C  = (Xz @ Yz.T) / (Tseg - 1)
    # matches = greedy_match_corr(C.copy())
    # if len(matches) == 0:
    #     ax_diag.text(0.5, 0.5, "No match", ha="center", va="center", transform=ax_diag.transAxes)
    # else:
    # i_best, j_best, _ = matches[0]
        # fixed choices
    i_best = j_best = 0
    if bool(cfg.get("timevary_delay", False)):
        # lag vs time
        t_idx, lags = windowed_best_lag(_z(sh1[i_best, :Tseg]), _z(sh2[j_best, :Tseg]),
                                        max_lag=12, win=60, step=10)
        ax_diag.plot(t_idx / fs, np.array(lags) * 1000 / fs, lw=1.1)
        # ax_diag.set_title("Best lag vs time", fontsize=9)
        ax_diag.set_xlabel("Time (s)"); ax_diag.set_ylabel("Lag (ms)")
        # ax_diag.yaxis.set_label_position("right"); 
        # ax_diag.yaxis.tick_right()
    else:
        # nonlinear relation (scatter + median curve)
        x = _z(sh1[i_best, :Tseg]); y = _z(sh2[j_best, :Tseg])
        # idx = np.linspace(0, len(x)-1, min(600, len(x))).astype(int)
        # ax_diag.plot(x[idx], y[idx], '.', ms=1.5, alpha=0.25)
        ax_diag.plot(x, y, '.', ms=1.5,color ="black" ,alpha=0.25)
        # nbins = 25
        # edges = np.linspace(x.min(), x.max(), nbins+1)
        # xc, ym = [], []
        # for b in range(nbins):
        #     m = (x>=edges[b]) & (x<edges[b+1])
        #     if m.sum() >= 5:
        #         xc.append(0.5*(edges[b]+edges[b+1]))
        #         ym.append(np.median(y[m]))
        # if len(xc):
        #     ax_diag.plot(xc, ym, lw=1.4)
        ax_diag.set_title("Nonlinear relation", fontsize=9)
        ax_diag.set_xlabel("shared region1 (z)"); ax_diag.set_ylabel("shared region2 (z)")
        # ax_diag.yaxis.set_label_position("right"); 
        # ax_diag.yaxis.tick_right()

    # cosmetics
    for ax in (ax_obs, ax_sh, ax_pr, ax_diag):
        for s in ("top","right"):
            ax.spines[s].set_visible(False)

# ---------- full figure driver for figure A.1 ----------
def plot_joint_latents_figure(ALL_DATASETS, trial_idx=0, secs=2.0,
                              outpath="synthetic_joint_overview.pdf",
                              latent_ylim=(-2.5, 2.0)):
    n = len(ALL_DATASETS)
    assert n == 3, "This layout assumes exactly three datasets (3 rows)."

    fig = plt.figure(figsize=(10.5, 8.0))
    gs = fig.add_gridspec(nrows=n, ncols=1, hspace=0.70)

    for r, ds in enumerate(ALL_DATASETS):
        name = ds.get("name", ds.get("regime", f"Dataset {r+1}"))
        outer_cell = gs[r, 0]
        render_dataset_row(fig, outer_cell, ds, trial_idx=trial_idx, secs=secs, latent_ylim=latent_ylim)

        # # put your row label on the left margin
        # fig.text(0.08, 0.89 - r*(0.89-0.10)/(n-1), name, fontsize=11, weight="bold", va="center", ha = "center", rotation = 90)

        # --- option 2: compute row center from gridspec cell ---
        tmp_ax = fig.add_subplot(outer_cell)      # dummy axis
        pos = tmp_ax.get_position()               # [x0,y0,x1,y1] in fig coords
        tmp_ax.remove()                           # remove dummy
        y_center = (pos.y0 + pos.y1) / 2

        # put row label at the vertical center of the row
        fig.text(0.06, y_center, name,
                 fontsize=11, weight="bold",
                 va="center", ha="center", rotation=90)

    # column headers
    fig.text(0.12, 0.97, "Observed channel 0", fontsize=11, weight="bold")
    fig.text(0.4, 0.97, "Example latents (dim 0)",           fontsize=11, weight="bold")
    fig.text(0.62, 0.97, "Relation of shared1 and shared2",              fontsize=11, weight="bold")
    fig.text(0.7, 0.3, "Lag vs time",              fontsize=11, weight="bold")

    # space on the right for outside legends
    fig.subplots_adjust(right=0.86, left=0.10, top=0.93, bottom=0.07)
    fig.savefig(outpath, bbox_inches="tight")
    print(f"Saved: {outpath}")

#__________________helpers for figure 2
def _to_numpy(arr):
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    return np.asarray(arr)

def _match_T_and_flatten(model_latents, gt_latents):
    """
    model_latents: (N, Tm, D)
    gt_latents   : (N, D, Tg)
    Returns:
      model_flat, gt_flat, Tm, D
    """
    model_latents = _to_numpy(model_latents)
    gt_latents    = _to_numpy(gt_latents)

    N, Tm, D = model_latents.shape
    # GT (N,D,Tg) -> (N,Tg,D) -> trim to last Tm
    gt_latents = np.transpose(gt_latents, (0, 2, 1))
    if gt_latents.shape[1] < Tm:
        raise ValueError(f"GT has shorter T ({gt_latents.shape[1]}) than model T ({Tm}).")
    gt_trim = gt_latents[:, -Tm:, :]   # (N, Tm, D)

    # Flatten across trials & time -> (N*Tm, D)
    model_flat = model_latents.reshape(-1, D)
    gt_flat    = gt_trim.reshape(-1, D)
    return model_flat, gt_flat, Tm, D, gt_trim

def _cca_align(model_latents, gt_latents):
    """
    Fit CCA on flattened (N*Tm,D), return aligned sequences back in (N,Tm,D) form.
    """
    model_flat, gt_flat, Tm, D, gt_trim_3d = _match_T_and_flatten(model_latents, gt_latents)

    # Zero-mean each set (helps numerical stability)
    model_flat_c = model_flat - model_flat.mean(axis=0, keepdims=True)
    gt_flat_c    = gt_flat    - gt_flat.mean(axis=0, keepdims=True)

    cca = CCA(n_components=D, scale=False, max_iter=5000, tol=1e-6)
    m_cca, g_cca = cca.fit_transform(model_flat_c, gt_flat_c)

    # Per-component Pearson r in CCA space
    cca_corrs = [np.corrcoef(m_cca[:, i], g_cca[:, i])[0, 1] for i in range(m_cca.shape[1])]

    # Reshape back to (N, Tm, D)
    N = model_latents.shape[0]
    m_cca_3d = m_cca.reshape(N, Tm, -1)
    g_cca_3d = g_cca.reshape(N, Tm, -1)

    return m_cca_3d, g_cca_3d, np.array(cca_corrs, dtype=float)

# ---------- full figure driver for figure 2----------
def plot_shared1_mean_sem(
    spire_shared1, dlag_shared1, gt_shared1,
    dims=(0, 1, 2),
    T_clip=None,
    width_in=6.9,            # total figure width (two-column default)
    row_height_in=1.0,       # height per row (inches)
    base_fontsize=8,
    line_width=1.1,
    savepath=None,
    dpi=300,
    panel_labels=True,
    legend_headroom=0.12,
):
    """Mean±SEM of shared-1 latents; layout = 2 rows x len(dims) cols.
       Row 0: SPIRE [Comp d1..]; Row 1: DLAG [Comp d1..].
    """
    # Colors / styles
    model_color = "crimson"
    gt_color    = "teal"
    model_style = "-"
    gt_style    = "--"
    band_alpha  = 0.20

    # --- Per-model CCA alignment to GT (independent) ---
    spire_m, spire_g, spire_ccas = _cca_align(spire_shared1, gt_shared1)
    dlag_m,  dlag_g,  dlag_ccas  = _cca_align(dlag_shared1,  gt_shared1)

    # --- Common time window ---
    T_common = min(spire_m.shape[1], dlag_m.shape[1])
    if T_clip is not None:
        T_common = min(T_common, int(T_clip))
    spire_m, spire_g = spire_m[:, :T_common, :], spire_g[:, :T_common, :]
    dlag_m,  dlag_g  = dlag_m[:,  :T_common, :], dlag_g[:,  :T_common, :]
    x = np.arange(T_common)

    # --- Valid dims across models ---
    D_min = min(spire_m.shape[2], dlag_m.shape[2])
    dims = tuple(d for d in dims if d < D_min)
    if not dims:
        raise ValueError(f"No valid dims to plot; min D across models is {D_min}.")

    # --- Mean ± SEM ---
    def mean_sem(a):
        N = a.shape[0]
        m = a.mean(axis=0)
        s = a.std(axis=0, ddof=1)/np.sqrt(N) if N > 1 else np.zeros_like(m)
        return m, s

    spire_m_mean, spire_m_sem = mean_sem(spire_m)
    spire_g_mean, spire_g_sem = mean_sem(spire_g)
    dlag_m_mean,  dlag_m_sem  = mean_sem(dlag_m)
    dlag_g_mean,  dlag_g_sem  = mean_sem(dlag_g)

    # --- Figure at final size: 2 rows x ncols ---
    ncols = len(dims)
    nrows = 2
    height_in = max(2, nrows * row_height_in)

    rc = {
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "text.usetex": False,
        "font.family": "serif",  # IMPORTANT so font.serif below is used
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L", "TeX Gyre Termes", "DejaVu Serif"],
        "mathtext.fontset": "stix",   # Times-like math; use "cm" only if you want CM on purpose
        "font.size": base_fontsize,
        "axes.labelsize": base_fontsize,
        "axes.titlesize": base_fontsize,
        "xtick.labelsize": max(base_fontsize-1, 5),
        "ytick.labelsize": max(base_fontsize-1, 5),
        "legend.fontsize": max(base_fontsize-1, 5),
        "lines.linewidth": line_width,
    }

    with mpl.rc_context(rc):
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                                 figsize=(width_in, height_in))
        if ncols == 1:
            axes = axes.reshape(nrows, 1)

        # Collect handles once for a shared legend
        handles, labels = [], []

        # fontdict for titles (keeps Times consistent)
        title_fd = {"family": mpl.rcParams["font.family"], "size": mpl.rcParams["axes.titlesize"]}

        for c, d in enumerate(dims):
            # ---------- Row 0: SPIRE ----------
            ax = axes[0, c]
            (h_mod,) = ax.plot(x, spire_m_mean[:, d], model_style, color=model_color, label="Model mean")
            ax.fill_between(x,
                            spire_m_mean[:, d]-spire_m_sem[:, d],
                            spire_m_mean[:, d]+spire_m_sem[:, d],
                            color=model_color, alpha=band_alpha, label="Model ±SEM")
            (h_gt,)  = ax.plot(x, spire_g_mean[:, d], gt_style, color=gt_color, label="GT mean")
            ax.fill_between(x,
                            spire_g_mean[:, d]-spire_g_sem[:, d],
                            spire_g_mean[:, d]+spire_g_sem[:, d],
                            color=gt_color, alpha=band_alpha, label="GT ±SEM")
            ax.grid(alpha=0.3); ax.set_xlim(x[0], x[-1])
            ax.set_title(f"SPIRE — Comp {d+1}", fontdict=title_fd, pad=2)
            if c == 0:
                handles, labels = [h_mod, h_gt], ["Model mean", "GT mean"]

            # ---------- Row 1: DLAG ----------
            ax = axes[1, c]
            ax.plot(x, dlag_m_mean[:, d], model_style, color=model_color)
            ax.fill_between(x,
                            dlag_m_mean[:, d]-dlag_m_sem[:, d],
                            dlag_m_mean[:, d]+dlag_m_sem[:, d],
                            color=model_color, alpha=band_alpha)
            ax.plot(x, dlag_g_mean[:, d], gt_style, color=gt_color)
            ax.fill_between(x,
                            dlag_g_mean[:, d]-dlag_g_sem[:, d],
                            dlag_g_mean[:, d]+dlag_g_sem[:, d],
                            color=gt_color, alpha=band_alpha)
            ax.grid(alpha=0.3); ax.set_xlim(x[0], x[-1])
            ax.set_title(f"DLAG — Comp {d+1}", fontdict=title_fd, pad=2)

            # x-labels only on bottom row
            axes[1, c].set_xlabel("Time (samples)")

        # Single figure-level y-label (no per-axis ylabels)
        for r in range(nrows):
            for c in range(ncols):
                axes[r, c].set_ylabel(None)
        fig.text(0.001, 0.5, "Aligned latent (a.u.)",
                 rotation="vertical", va="center", ha="left")

        # Panel tags (a–f) across rows, left→right then next row
        if panel_labels:
            import string
            tags = iter(string.ascii_lowercase)
            for r in range(nrows):
                for c in range(ncols):
                    axes[r, c].text(0.02, 0.95, f"({next(tags)})",
                                    transform=axes[r, c].transAxes,
                                    ha="left", va="top",
                                    fontsize=base_fontsize, weight="bold")

        # Reserve a top band for the legend
        top_rect = 1.0 - legend_headroom
        plt.tight_layout(rect=[0.0, 0.0, 1.0, top_rect])

        # Legend (Matplotlib way), with font
        from matplotlib.font_manager import FontProperties
        legend_prop = FontProperties(family="Times New Roman", size=max(base_fontsize-1, 5))
        legend_y = top_rect + legend_headroom/2.0
        fig.legend(handles, labels, ncol=2, loc="center",
                   bbox_to_anchor=(0.5, legend_y), frameon=False, prop=legend_prop)

        if savepath:
            fig.savefig(savepath, bbox_inches="tight", dpi=dpi)

    return fig, axes, {"SPIRE_cca": spire_ccas, "DLAG_cca": dlag_ccas}