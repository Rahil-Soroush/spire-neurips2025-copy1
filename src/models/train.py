#training loop
import numpy as np
import time
import os

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler #speed up using mixed precision
from torch.utils.tensorboard import SummaryWriter

from models.spire_model import SPIREAutoencoder
from utils.losses import reconstruction_loss, orthogonality_loss, vicreg_loss
import utils.training_utils as tr_util


def train_spire_synth(X_tensor, y_tensor, shared_dim=3, private_dim=3, hidden_dim=64, dropout_prob=0.3, num_epochs=200, batch_size=8, run_name="test1", patience=20,
    model_save_path="best_model.pt",FREEZE_BEG=90, FREEZE_END = 110,disable_private_ramp=False,identity_aligner = False,var_guards=True,**weights):
    """
    Trains the SPIRE model on paired GPi-STN segments with alignment and orthogonality constraints.

    Args:
        X_tensor (Tensor): Input tensor from GPi (segments, time, channels)
        y_tensor (Tensor): Input tensor from STN (segments, time, channels)
        ...
    Returns:
        model: Trained SPIRE model (best checkpoint)
        val_loader: DataLoader for validation set
        device: torch.device used (cuda or cpu)
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1)

    model = SPIREAutoencoder(
        input_dim_gpi=X_tensor.shape[2],#number of channels
        input_dim_stn=y_tensor.shape[2],
        shared_dim=shared_dim,
        private_dim=private_dim,
        hidden_dim=hidden_dim,dropout_prob=dropout_prob
    ).to(device)

    model.use_identity_align = bool(identity_aligner)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    writer = SummaryWriter(log_dir="runs/SPIRE_synthData_" + run_name)
    scaler = GradScaler()#Prevents underflow during backprop

    # which terms are ON this run? (presence in **weights toggles inclusion)
    include = {k: (k in weights) for k in ["w_rec","w_align","w_orth",
                                       "w_cross","w_self",
                                       "w_mapid","w_align_reg"]}

    best_val_loss = float('inf')
    epochs_no_improve = 0
    start = time.time()

    for epoch in range(num_epochs):
        # numeric values are treated as scale multipliers
        scales = {k: float(v) for k, v in weights.items() if isinstance(v, (int, float))}
        alpha_ramp_end = 140 # will be used for preventing early stopping as well
        w = tr_util.sched_synth(epoch, include, scales=scales,
                          ramp_start=20, ramp_end=100, 
                          alpha_ramp_start=80, alpha_ramp_end=alpha_ramp_end)  

        if disable_private_ramp:   # add an arg to train_spire_autoencoder, default False
            w["alpha_p"] = 1.0

        alpha_p = w["alpha_p"]
        w["w_orth"] *= max(w["alpha_p"], 1e-3)   # gentle ramp; keeps a floor
        if w["alpha_p"] > 0.5:
            w["w_cross"] *= 1.5   # or clamp to ~0.10 max

        # optional freeze window
        if (FREEZE_BEG is not None) and epoch == FREEZE_BEG: tr_util.freeze_shared_modules(model)
        if (FREEZE_END is not None) and epoch == FREEZE_END: tr_util.unfreeze_shared_modules(model)
        freeze_active = (FREEZE_BEG is not None and FREEZE_END is not None and FREEZE_BEG <= epoch < FREEZE_END)

        # reduce, don't kill, alignment pressure while frozen
        if freeze_active and w["w_align"] > 0:
            w["w_align"] *= 0.3

        model.train()
        total_loss, total_recon_loss, total_align_loss, total_orth_loss  = 0, 0, 0, 0
        total_cos_sim = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)

            with autocast():
                recon_xb, recon_yb, shared_xb, shared_yb, private_xb, private_yb = model(xb, yb, private_gate=alpha_p)

                if freeze_active:
                    shared_xb = shared_xb.detach(); shared_yb = shared_yb.detach()

                # aligner paths (mapper-only if identity_aligner=True)
                need_align = (w["w_align"]>0) or (w["w_cross"]>0)
                if need_align:
                    zx_y = model.align_x_to_y(shared_xb)   # x->y
                    zy_x = model.align_y_to_x(shared_yb)   # y->x
                else:
                    zx_y, zy_x = shared_xb, shared_yb

                # losses
                loss_recon = reconstruction_loss(xb, recon_xb, yb, recon_yb) if w["w_rec"]>0 else torch.zeros((), device=device)

                loss_self = torch.zeros((), device=device)
                if w["w_self"]>0:
                    x_s = model.decoder_gpi(shared_xb, torch.zeros_like(private_xb))
                    y_s = model.decoder_stn(shared_yb, torch.zeros_like(private_yb))
                    loss_self = F.mse_loss(x_s, xb) + F.mse_loss(y_s, yb)

                loss_cross = torch.zeros((), device=device)
                if w["w_cross"]>0:
                    x_from_y = model.decoder_gpi(zy_x, torch.zeros_like(private_yb))
                    y_from_x = model.decoder_stn(zx_y, torch.zeros_like(private_xb))
                    loss_cross = F.mse_loss(x_from_y, xb) + F.mse_loss(y_from_x, yb)

                loss_align =  torch.zeros((), device=device)
                if w["w_align"]>0:
                    loss_align = 0.5*(vicreg_loss(shared_xb, zy_x) + vicreg_loss(shared_yb, zx_y))

                loss_orth =  torch.zeros((), device=device)
                if w["w_orth"]>0:
                    loss_orth = orthogonality_loss(tr_util.norm_latent(shared_xb), tr_util.norm_latent(private_xb)) + \
                                orthogonality_loss(tr_util.norm_latent(shared_yb), tr_util.norm_latent(private_yb))

                L_mapid = model.mapper_identity_loss() if w["w_mapid"]>0 else torch.zeros((), device=device)
                L_align_reg = (model.align_regularizer() if w["w_align_reg"]>0  else torch.zeros((), device=device))

                # variance guards
                w_var_sh = 5e-3
                L_var_sh = tr_util.variance_guard(shared_xb) + tr_util.variance_guard(shared_yb)
                w_var_pf = 2e-3
                L_var_pf = tr_util.variance_floor(private_xb, 0.20) + tr_util.variance_floor(private_yb, 0.20)

                loss = (w["w_rec"]*loss_recon + w["w_align"]*loss_align + w["w_orth"]*loss_orth +
                        w["w_cross"]*loss_cross + w["w_self"]*loss_self +
                        w["w_mapid"]*L_mapid + w["w_align_reg"]*L_align_reg)
                if var_guards:
                    loss = loss +w_var_sh*L_var_sh + w_var_pf*L_var_pf
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()

            total_loss   += float(loss.detach())
            total_recon_loss   += float(loss_recon.detach())
            total_align_loss += float(loss_align.detach())
            total_orth_loss  += float(loss_orth.detach())

            # cosine sim for monitoring
            with torch.no_grad():
                cos_sim = F.cosine_similarity(
                    F.normalize(shared_xb, dim=-1, eps=1e-8),
                    F.normalize(shared_yb, dim=-1, eps=1e-8),
                    dim=-1, eps=1e-8
                ).mean()
                total_cos_sim += float(cos_sim.mean())

        avg_loss = total_loss / len(train_loader)
        avg_recon = total_recon_loss / len(train_loader)
        avg_align = total_align_loss / len(train_loader)
        avg_cos_sim = total_cos_sim / len(train_loader)
        avg_orth = total_orth_loss / len(train_loader)


        # --- Validation ---
        model.eval()
        val_loss, val_recon_loss, val_align_loss, val_orth_loss = 0, 0, 0, 0
        val_cos_sim = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                recon_xb, recon_yb, zx, zy, px, py = model(xb, yb, private_gate=alpha_p)
                need_align = (w["w_align"]>0) or (w["w_cross"]>0)
                if need_align:
                    zx_y = model.align_x_to_y(zx)
                    zy_x = model.align_y_to_x(zy)
                else:
                    zx_y, zy_x = zx, zy

                loss_recon = reconstruction_loss(xb, recon_xb, yb, recon_yb) if w["w_rec"]>0 else 0.0

                loss_self  = 0.0
                if w["w_self"]>0:
                    x_s = model.decoder_gpi(zx, torch.zeros_like(px))
                    y_s = model.decoder_stn(zy, torch.zeros_like(py))
                    loss_self = F.mse_loss(x_s, xb) + F.mse_loss(y_s, yb)

                loss_cross = 0.0
                if w["w_cross"]>0:
                    x_from_y = model.decoder_gpi(zy_x, torch.zeros_like(py))
                    y_from_x = model.decoder_stn(zx_y, torch.zeros_like(px))
                    loss_cross = F.mse_loss(x_from_y, xb) + F.mse_loss(y_from_x, yb)

                loss_align = 0.0
                if w["w_align"]>0:
                    loss_align = 0.5*(vicreg_loss(zx, zy_x) + vicreg_loss(zy, zx_y))

                loss_orth = 0.0
                if w["w_orth"]>0:
                    loss_orth = orthogonality_loss(tr_util.norm_latent(zx), tr_util.norm_latent(px)) + \
                                orthogonality_loss(tr_util.norm_latent(zy), tr_util.norm_latent(py))
                    
                L_mapid = model.mapper_identity_loss() if w["w_mapid"]>0 else 0.0
                L_align_reg = (model.align_regularizer() if w["w_align_reg"]>0  else 0.0)

                w_var_sh = 5e-3
                L_var_sh = tr_util.variance_guard(zx) + tr_util.variance_guard(zy)
                w_var_pf = 2e-3
                L_var_pf = tr_util.variance_floor(px, 0.20) + tr_util.variance_floor(py, 0.20)

                vloss = (w["w_rec"]*loss_recon + w["w_align"]*loss_align + w["w_orth"]*loss_orth +
                         w["w_cross"]*loss_cross + w["w_self"]*loss_self +
                         w["w_mapid"]*L_mapid + w["w_align_reg"]*L_align_reg)
                if var_guards:
                    vloss = vloss +w_var_sh*L_var_sh + w_var_pf*L_var_pf

                val_loss   += float(vloss)
                val_recon_loss   += float(loss_recon if isinstance(loss_recon, (float,int)) else loss_recon.item())
                val_align_loss += float(loss_align if isinstance(loss_align, (float,int)) else loss_align.item())
                val_orth_loss  += float(loss_orth if isinstance(loss_orth, (float,int)) else loss_orth.item())

                cos_sim = F.cosine_similarity(
                    F.normalize(shared_xb, dim=-1),
                    F.normalize(shared_yb, dim=-1),
                    dim=-1
                )
                val_cos_sim += float(cos_sim.mean())

        avg_val_loss = val_loss / len(val_loader)
        avg_val_recon = val_recon_loss / len(val_loader)
        avg_val_align = val_align_loss / len(val_loader)
        avg_val_cos_sim = val_cos_sim / len(val_loader)
        avg_val_orth = val_orth_loss / len(val_loader)

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        writer.add_scalar("Loss/train_total", avg_loss, epoch)
        writer.add_scalar("Loss/train_recon", avg_recon, epoch)
        writer.add_scalar("Loss/train_align", avg_align, epoch)
        writer.add_scalar("Loss/train_orth", avg_orth, epoch)

        writer.add_scalar("Loss/val_total", avg_val_loss, epoch)
        writer.add_scalar("Loss/val_recon", avg_val_recon, epoch)
        writer.add_scalar("Loss/val_align", avg_val_align, epoch)
        writer.add_scalar("Loss/val_orth", avg_val_orth, epoch)

        writer.add_scalar("CosineSimilarity/train", avg_cos_sim, epoch)
        writer.add_scalar("CosineSimilarity/val", avg_val_cos_sim, epoch)
        writer.add_scalar("LR", current_lr, epoch)

        writer.add_scalar("PrivateGate/alpha_p", alpha_p, epoch)
        writer.add_scalar("Aligner/identity_mode", 1.0 if model.use_identity_align else 0.0, epoch)
        
        # #checking
        # if epoch % 10 == 0 or epoch == num_epochs - 1:
        #     print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

        # --- Early Stopping & Checkpoint ---
        if epoch in (119, 140, 170, 199):
            torch.save(model.state_dict(), model_save_path.replace(".pt", f"_e{epoch}.pt"))
            # If you want an auxiliary bundle, write it to a different file:
            aux_path = model_save_path.replace(".pt", f"_e{epoch}_bundle.pth")
            torch.save({
                'model': model,
                'val_set': val_set
            }, aux_path)

        # ES
        IGNORE_ES_UNTIL = alpha_ramp_end  # == alpha_ramp_end
        if epoch >= IGNORE_ES_UNTIL:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss; epochs_no_improve = 0
                torch.save(model.state_dict(), model_save_path)
                print(f"✅ New best @ epoch {epoch+1} val={best_val_loss:.4f}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"⏹️ Early stopping @ epoch {epoch+1}")
                    break
    
    end = time.time()
    print(f"Training took {(end - start)/60:.2f} minutes")
    # Load the best model before returning
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    aux_path = model_save_path.replace(".pt", "_bundle.pth")
    torch.save({
        'model': model,
        'val_set': val_set
    }, aux_path)

    # ensure TB events are flushed
    writer.close()

    return model, val_loader, device

def train_spire_real(X_tensor, y_tensor, shared_dim=3, private_dim=3, hidden_dim=64, dropout_prob=0.2, num_epochs=200, batch_size=8,run_name="test1",patience=20,
    model_save_path="best_model.pt",FREEZE_BEG=None, FREEZE_END=None, identity_aligner=True, weight_schedule=tr_util.sched_real,**weights):
    """
    Trains the SPIRE model on paired GPi-STN segments with alignment and orthogonality constraints.

    Args:
        X_tensor (Tensor): Input tensor from GPi (segments, time, channels)
        y_tensor (Tensor): Input tensor from STN (segments, time, channels)
        ...
    Returns:
        model: Trained SPIRE model (best checkpoint)
        val_loader: DataLoader for validation set
        device: torch.device used (cuda or cpu)
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1)

    model = SPIREAutoencoder(
        input_dim_gpi=X_tensor.shape[2],#number of channels
        input_dim_stn=y_tensor.shape[2],
        shared_dim=shared_dim,
        private_dim=private_dim,
        hidden_dim=hidden_dim,dropout_prob=dropout_prob
    ).to(device)

    # warm start: mapper-only aligner, then enable conv-align after epoch 60
    model.use_identity_align = bool(identity_aligner)
    identity_off_epoch = 60

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    writer = SummaryWriter(log_dir="runs/SPIRE_realData_" + run_name)
    scaler = GradScaler()#Prevents underflow during backprop

    best_val = float('inf')
    no_improve = 0
    IGNORE_ES_UNTIL = 140   # let schedule settle before ES

    for epoch in range(num_epochs):
        if model.use_identity_align and (epoch >= identity_off_epoch):
            model.use_identity_align = False

        w = weight_schedule(epoch)
        # ensure all keys exist
        for k in ["w_rec","w_align","w_orth","w_cross","w_self","w_mapid","w_align_reg","alpha_p"]:
            w.setdefault(k, 0.0)
        alpha_p = float(np.clip(w["alpha_p"], 0.0, 1.0))
        private_gate_eff = max(alpha_p, 0.10)

        # optional freeze window
        if (FREEZE_BEG is not None) and epoch == FREEZE_BEG: tr_util.freeze_shared_modules(model)
        if (FREEZE_END is not None) and epoch == FREEZE_END: tr_util.unfreeze_shared_modules(model)
        freeze_active = (FREEZE_BEG is not None and FREEZE_END is not None and FREEZE_BEG <= epoch < FREEZE_END)

        model.train()
        total = 0.0
        total_recon_loss = total_align_loss = total_orth_loss = total_cos_sim = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)

            with autocast():
                recon_xb, recon_yb, shared_xb, shared_yb, private_xb, private_yb = model(xb, yb, private_gate=private_gate_eff)

                if freeze_active:
                    shared_xb = shared_xb.detach(); shared_yb = shared_yb.detach()

                # aligner paths (mapper-only if identity_aligner=True)
                need_align = (w["w_align"]>0) or (w["w_cross"]>0)
                if need_align:
                    zx_y = model.align_x_to_y(shared_xb)   # x->y
                    zy_x = model.align_y_to_x(shared_yb)   # y->x
                else:
                    zx_y, zy_x = shared_xb, shared_yb

                # losses
                loss_recon = reconstruction_loss(xb, recon_xb, yb, recon_yb) if w["w_rec"]>0 else torch.zeros((), device=device)

                loss_self = torch.zeros((), device=device)
                if w["w_self"]>0:
                    x_s = model.decoder_gpi(shared_xb, torch.zeros_like(private_xb))
                    y_s = model.decoder_stn(shared_yb, torch.zeros_like(private_yb))
                    loss_self = F.mse_loss(x_s, xb) + F.mse_loss(y_s, yb)

                loss_cross = torch.zeros((), device=device)
                if w["w_cross"]>0:
                    x_from_y = model.decoder_gpi(zy_x, torch.zeros_like(private_yb))
                    y_from_x = model.decoder_stn(zx_y, torch.zeros_like(private_xb))
                    loss_cross = F.mse_loss(x_from_y, xb) + F.mse_loss(y_from_x, yb)

                loss_align = torch.zeros((), device=device)
                if w["w_align"]>0:
                    loss_align = 0.5*(vicreg_loss(shared_xb, zy_x) + vicreg_loss(shared_yb, zx_y))

                loss_orth = torch.zeros((), device=device)
                if w["w_orth"]>0:
                    loss_orth = orthogonality_loss(tr_util.norm_latent(shared_xb), tr_util.norm_latent(private_xb)) + \
                                orthogonality_loss(tr_util.norm_latent(shared_yb), tr_util.norm_latent(private_yb))

                L_mapid = model.mapper_identity_loss() if w["w_mapid"]>0 else torch.zeros((), device=device)
                L_align_reg = (model.align_regularizer() if (w["w_align_reg"]>0 and not model.use_identity_align) else torch.zeros((), device=device))

                # variance guards
                w_var_sh = 5e-3
                L_var_sh = tr_util.variance_guard(shared_xb) + tr_util.variance_guard(shared_yb)
                w_var_pf = 2e-3
                L_var_pf = tr_util.variance_floor(private_xb, 0.20) + tr_util.variance_floor(private_yb, 0.20)

                loss = (w["w_rec"]*loss_recon + w["w_align"]*loss_align + w["w_orth"]*loss_orth +
                        w["w_cross"]*loss_cross + w["w_self"]*loss_self +
                        w["w_mapid"]*L_mapid + w["w_align_reg"]*L_align_reg +
                        w_var_sh*L_var_sh + w_var_pf*L_var_pf)
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()

            total += float(loss.detach())
            total_recon_loss   += float(loss_recon.detach())
            total_align_loss += float(loss_align.detach())
            total_orth_loss  += float(loss_orth.detach())

            # cosine sim for monitoring
            with torch.no_grad():
                cos_sim = F.cosine_similarity(
                    F.normalize(shared_xb, dim=-1, eps=1e-8),
                    F.normalize(shared_yb, dim=-1, eps=1e-8),
                    dim=-1, eps=1e-8
                ).mean()
                total_cos_sim += float(cos_sim.mean())

        avg_loss = total / len(train_loader)
        avg_recon = total_recon_loss / len(train_loader)
        avg_align = total_align_loss / len(train_loader)
        avg_cos_sim = total_cos_sim / len(train_loader)
        avg_orth = total_orth_loss / len(train_loader)


        # --- Validation ---
        model.eval()
        vtotal = 0.0
        val_recon_loss= val_align_loss= val_orth_loss = val_self_loss= val_cross_loss = 0.0
        val_cos_sim = 0.0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                recon_xb, recon_yb, zx, zy, px, py = model(xb, yb, private_gate=private_gate_eff)
                need_align = (w["w_align"]>0) or (w["w_cross"]>0)
                if need_align:
                    zx_y = model.align_x_to_y(zx)
                    zy_x = model.align_y_to_x(zy)
                else:
                    zx_y, zy_x = zx, zy

                loss_recon = reconstruction_loss(xb, recon_xb, yb, recon_yb) if w["w_rec"]>0 else 0.0
                loss_self  = 0.0
                if w["w_self"]>0:
                    x_s = model.decoder_gpi(zx, torch.zeros_like(px))
                    y_s = model.decoder_stn(zy, torch.zeros_like(py))
                    loss_self = F.mse_loss(x_s, xb) + F.mse_loss(y_s, yb)
                loss_cross = 0.0
                if w["w_cross"]>0:
                    x_from_y = model.decoder_gpi(zy_x, torch.zeros_like(py))
                    y_from_x = model.decoder_stn(zx_y, torch.zeros_like(px))
                    loss_cross = F.mse_loss(x_from_y, xb) + F.mse_loss(y_from_x, yb)
                loss_align = 0.0
                if w["w_align"]>0:
                    loss_align = 0.5*(vicreg_loss(zx, zy_x) + vicreg_loss(zy, zx_y))
                loss_orth = 0.0
                if w["w_orth"]>0:
                    loss_orth = orthogonality_loss(tr_util.norm_latent(zx), tr_util.norm_latent(px)) + \
                                orthogonality_loss(tr_util.norm_latent(zy), tr_util.norm_latent(py))
                L_mapid = model.mapper_identity_loss() if w["w_mapid"]>0 else 0.0
                L_align_reg = (model.align_regularizer() if (w["w_align_reg"]>0 and not model.use_identity_align) else 0.0)

                w_var_sh = 5e-3
                L_var_sh = tr_util.variance_guard(zx) + tr_util.variance_guard(zy)
                w_var_pf = 2e-3
                L_var_pf = tr_util.variance_floor(px, 0.20) + tr_util.variance_floor(py, 0.20)

                vloss = (w["w_rec"]*loss_recon + w["w_align"]*loss_align + w["w_orth"]*loss_orth +
                         w["w_cross"]*loss_cross + w["w_self"]*loss_self +
                         w["w_mapid"]*L_mapid + w["w_align_reg"]*L_align_reg +
                         w_var_sh*L_var_sh + w_var_pf*L_var_pf)
                vtotal += float(vloss)
                val_recon_loss   += float(loss_recon if isinstance(loss_recon, (float,int)) else loss_recon.item())
                val_align_loss += float(loss_align if isinstance(loss_align, (float,int)) else loss_align.item())
                val_orth_loss  += float(loss_orth if isinstance(loss_orth, (float,int)) else loss_orth.item())
                val_self_loss += float(loss_self if isinstance(loss_self, (float,int)) else loss_self.item())
                val_cross_loss += float(loss_cross if isinstance(loss_cross, (float,int)) else loss_cross.item())

        avg_v = vtotal / len(val_loader)
        avg_val_recon = val_recon_loss / len(val_loader)
        avg_val_align = val_align_loss / len(val_loader)
        avg_val_cos_sim = val_cos_sim / len(val_loader)
        avg_val_orth = val_orth_loss / len(val_loader)
        avg_val_self = val_self_loss / len(val_loader)
        avg_val_cross = val_cross_loss / len(val_loader)

        scheduler.step(avg_v)
        current_lr = optimizer.param_groups[0]['lr']

        writer.add_scalar("Loss/train_total", avg_loss, epoch)
        writer.add_scalar("Loss/train_recon", avg_recon, epoch)
        writer.add_scalar("Loss/train_align", avg_align, epoch)
        writer.add_scalar("Loss/train_orth", avg_orth, epoch)

        writer.add_scalar("Loss/val_total", avg_v, epoch)
        writer.add_scalar("Loss/val_recon", avg_val_recon, epoch)
        writer.add_scalar("Loss/val_align", avg_val_align, epoch)
        writer.add_scalar("Loss/val_orth", avg_val_orth, epoch)
        writer.add_scalar("Loss/val_self_shared", avg_val_self, epoch)
        writer.add_scalar("Loss/val_cross_shared", avg_val_cross, epoch)

        writer.add_scalar("CosineSimilarity/train", avg_cos_sim, epoch)
        writer.add_scalar("CosineSimilarity/val", avg_val_cos_sim, epoch)
        writer.add_scalar("LR", current_lr, epoch)
        
        # #checking
        # if epoch % 10 == 0 or epoch == num_epochs - 1:
        #     print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

        # --- Early Stopping & Checkpoint ---
        if epoch in (119, 140, 170, 199):
            torch.save(model.state_dict(), model_save_path.replace(".pt", f"_e{epoch}.pt"))
            # If you want an auxiliary bundle, write it to a different file:
            aux_path = model_save_path.replace(".pt", f"_e{epoch}_bundle.pth")
            torch.save({
                'model': model,
                'val_set': val_set
            }, aux_path)
        # ES
        if epoch >= IGNORE_ES_UNTIL:
            if avg_v < best_val:
                best_val = avg_v; no_improve = 0
                torch.save(model.state_dict(), model_save_path)
                print(f"✅ New best @ epoch {epoch+1} val={avg_v:.4f}")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"⏹️ Early stopping @ epoch {epoch+1}")
                    break

    # Load the best model before returning
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    aux_path = model_save_path.replace(".pt", "_bundle.pth")
    torch.save({
        'model': model,
        'val_set': val_set
    }, aux_path)
    return model, val_loader, device


