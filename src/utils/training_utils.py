#other helpers for training
import numpy as np
import random
import os
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def _cosine_ramp(epoch, start=5, end=60):
    """Smoothly ramp 0→1 between [start, end]."""
    if epoch <= start: return 0.0
    if epoch >= end:   return 1.0
    x = (epoch - start) / (end - start)
    return 0.5 * (1 - np.cos(np.pi * x))

def norm_latent(z, eps=1e-6):
    # z: (B,T,D)
    m = z.mean(dim=(0,1), keepdim=True)
    # s = z.std(dim=(0,1), keepdim=True).clamp_min(eps)
    # unbiased=False to avoid huge std spikes at small effective N; clamp floor
    s = z.std(dim=(0,1), unbiased=False, keepdim=True).clamp_min(eps)
    return (z - m) / s

#functions for freezing the shared pathways
def set_requires_grad(module, flag: bool):
    if module is None: 
        return
    for p in module.parameters():
        p.requires_grad_(flag)

def freeze_shared_modules(model, *, freeze_aligners: bool = False):
    """
    Freeze only the *encoder* shared projections so shared latents stay fixed.
    By default, keep aligners (ConvAlign/SoftShift) + linear mappers trainable
    so they can keep improving alignment while private ramps.
    """
    # encoders
    if hasattr(model, "encoder_gpi") and hasattr(model.encoder_gpi, "shared_proj"):
        set_requires_grad(model.encoder_gpi.shared_proj, False)
    if hasattr(model, "encoder_stn") and hasattr(model.encoder_stn, "shared_proj"):
        set_requires_grad(model.encoder_stn.shared_proj, False)

    # keep private heads ON
    if hasattr(model, "encoder_gpi") and hasattr(model.encoder_gpi, "private_proj"):
        set_requires_grad(model.encoder_gpi.private_proj, True)
    if hasattr(model, "encoder_stn") and hasattr(model.encoder_stn, "private_proj"):
        set_requires_grad(model.encoder_stn.private_proj, True)

    for name in ("align_x2y", "align_y2x"):
        if hasattr(model, name):
            set_requires_grad(getattr(model, name), not freeze_aligners)
    # legacy soft shift (for safety if still present)
    if hasattr(model, "shift_x"):
        set_requires_grad(model.shift_x, not freeze_aligners)
    if hasattr(model, "shift_y"):
        set_requires_grad(model.shift_y, not freeze_aligners)

    # linear mappers stay ON
    if hasattr(model, "map_x2y"):
        set_requires_grad(model.map_x2y, True)
    if hasattr(model, "map_y2x"):
        set_requires_grad(model.map_y2x, True)


def unfreeze_shared_modules(model):
    """Unfreeze encoder shared projections again."""
    if hasattr(model, "encoder_gpi") and hasattr(model.encoder_gpi, "shared_proj"):
        set_requires_grad(model.encoder_gpi.shared_proj, True)
    if hasattr(model, "encoder_stn") and hasattr(model.encoder_stn, "shared_proj"):
        set_requires_grad(model.encoder_stn.shared_proj, True)

def variance_floor(z_bt_d, min_std=0.25, eps=1e-6):
    """
    Hinge loss: penalize only if per-dim std < min_std.
    """
    z = z_bt_d.reshape(-1, z_bt_d.size(-1))          # (N, D)
    std = z.std(dim=0) + eps
    gap = (min_std - std).clamp_min(0.0)             # ReLU(min_std - std)
    return (gap**2).mean()

def variance_guard(z_bt_d, target=1.0, eps=1e-6):
    """
    Symmetric guard (used for SHARED), encourages std ≈ target.
    """
    z = z_bt_d.reshape(-1, z_bt_d.size(-1))
    std = z.std(dim=0) + eps
    return ((std - target)**2).mean()

def sched_real(epoch: int):
    if epoch < 60:     # warm: mapper-only, aligner identity, shared forming
        return dict(w_rec=1.0, w_align=0.30, w_orth=0.012,
                    w_self=0.03, w_cross=0.00,
                    w_mapid=0.005, w_align_reg=0.0,
                    alpha_p=1.0)
    elif epoch < 100:  # aligner ON, keep align modest
        return dict(w_rec=1.0, w_align=0.30, w_orth=0.012,
                    w_self=0.05, w_cross=0.05,
                    w_mapid=0.005, w_align_reg=5e-5,
                    alpha_p=1.0)
    elif epoch < 140:  # recover alignment, keep self/cross steady
        return dict(w_rec=1.0, w_align=0.38, w_orth=0.015,
                    w_self=0.05, w_cross=0.06,
                    w_mapid=0.003, w_align_reg=7.5e-5,
                    alpha_p=1.0)
    else:              # tighten alignment a bit more; mapper penalty off
        return dict(w_rec=1.0, w_align=0.45, w_orth=0.015,
                    w_self=0.05, w_cross=0.06,
                    w_mapid=0.0,  w_align_reg=1e-4,
                    alpha_p=1.0)
    

def sched_synth(epoch, include, scales=None,
                          ramp_start=20, ramp_end=100, 
                          alpha_ramp_start=80, alpha_ramp_end=140):
    # Phase schedule used for synthetic data
    scales = scales or {}
    phase = "pre" if epoch < ramp_start else ("ramp" if epoch < ramp_end else "post")

    # --- private gate ramp (Phase B) ---
    alpha_p = _cosine_ramp(epoch, alpha_ramp_start, alpha_ramp_end)  # NEW

    def on(k): return include.get(k, False)
    def sc(k): return float(scales.get(k, 1.0))
    def pick(k, v_pre, v_ramp, v_post):
        if not on(k): return 0.0
        v = {"pre": v_pre, "ramp": v_ramp, "post": v_post}[phase]
        return sc(k) * v

    w = {
        "w_rec":   1.0 if on("w_rec") else 0.0,
        "w_align": pick("w_align", 0.22, 0.10, 0.08),
        "w_orth":  pick("w_orth",  0.008, 0.015, 0.025),

        # keep small cross on
        "w_cross": pick("w_cross", 0.03, 0.05, 0.07),

        "w_self": pick("w_self",  0.02, 0.04, 0.03),

        # mapper penalty decays to zero (same as Phase A)
        "w_mapid": pick("w_mapid", 0.01, 0.005, 0.00),

        #aligner regularizer
        "w_align_reg": pick("w_align_reg", 1e-4, 5e-4, 1e-4),

        "alpha_p": alpha_p,
    }
    return w

#model ablation variants for synthetic data evaluation
def gen_ablation_variants(base):
    variants = {"SPIRE_synth": dict(base)}  # the reference

    # --- single-term ablations
    for k in ["w_align","w_orth","w_cross","w_self","w_align_reg","w_mapid"]:
        v = dict(base); v[k] = 0.0
        variants[f"abl_no_{k}"] = v

    # --- family ablations
    v = dict(base); v["w_align"] = 0.0; v["w_align_reg"] = 0.0
    variants["abl_no_alignment_family"] = v

    v = dict(base); v["w_orth"] = 0.0; v["w_mapid"] = 0.0
    variants["abl_no_disentanglement_family"] = v

    v = dict(base); v["w_orth"] = 0.0; v["w_self"] = 0.0
    variants["abl_shared_only_family"] = v

    v = dict(base); v["w_mapid"] = 0.0; v["w_align_reg"] = 0.0
    variants["abl_no_regularizers"] = v

    return variants
