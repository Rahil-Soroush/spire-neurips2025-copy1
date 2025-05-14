#training loop
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler #speed up using mixed precision
from torch.utils.tensorboard import SummaryWriter

from src.models.spire_model import SPIRE, SPIRE_3Region
from src.utils.losses import reconstruction_loss, cos_sim_loss, MSE_loss, orthogonality_loss, reconstruction_loss3



def train_spire(X_tensor, y_tensor, shared_dim=32, private_dim=32, hidden_dim=64, dropout_prob=0.3, num_epochs=200, batch_size=8,run_name="test1",patience=20,
    model_save_path="best_model.pt"):
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

    model = SPIRE(
        input_dim_gpi=X_tensor.shape[2],#number of channels
        input_dim_stn=y_tensor.shape[2],
        shared_dim=shared_dim,
        private_dim=private_dim,
        hidden_dim=hidden_dim,dropout_prob=dropout_prob
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    writer = SummaryWriter(log_dir="runs/SPIRE_DualAE_" + run_name)
    scaler = GradScaler()#Prevents underflow during backprop

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss, total_recon_loss, total_align_loss, total_orth_loss  = 0, 0, 0, 0
        total_cos_sim = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            # 🚀 Mixed precision forward + loss, Faster training, lower memory usage
            with autocast():
                recon_xb, recon_yb, shared_xb, shared_yb, private_xb, private_yb = model(xb, yb)
                loss_recon = reconstruction_loss(xb, recon_xb, yb, recon_yb)
                recon_shared_x = model.decoder_gpi(shared_xb, torch.zeros_like(private_xb))
                recon_shared_y = model.decoder_stn(shared_yb, torch.zeros_like(private_yb))
                loss_shared_recon = F.mse_loss(recon_shared_x, xb) + F.mse_loss(recon_shared_y, yb)
                
                loss_orth = orthogonality_loss(shared_xb, private_xb) + orthogonality_loss(shared_yb, private_yb)
                # loss = loss_recon + 0.3 * loss_align + 0.1 * loss_orth  # Feel free to tune this weight

                loss_align_mse = MSE_loss(shared_xb, shared_yb)
                loss_align_cos = cos_sim_loss(shared_xb, shared_yb)
                loss_align = 0.7 * loss_align_mse + 0.3 * loss_align_cos
                #loss = loss_recon + 0.4 * loss_align + 0.01 * loss_orth
                loss = loss_recon + 0.4 * loss_align + 0.01 * loss_orth + 0*loss_shared_recon#default was loss_recon + 0.1 * loss_align, a balance between:Preserving high reconstruction accuracy,Encouraging meaningful shared structure

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            total_loss += loss.item()
            total_recon_loss += loss_recon.item()
            total_align_loss += loss_align.item()
            total_orth_loss += loss_orth.item()

            # 🔥 Cosine similarity between shared latents (mean over batch & time)
            cos_sim = F.cosine_similarity(
                F.normalize(shared_xb, dim=-1),
                F.normalize(shared_yb, dim=-1),
                dim=-1
            )  # shape: [B, T]
            total_cos_sim += cos_sim.mean().item()

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
                recon_xb, recon_yb, shared_xb, shared_yb, private_xb, private_yb = model(xb, yb)
                loss_recon = reconstruction_loss(xb, recon_xb, yb, recon_yb)
                #shared only recon:
                recon_shared_x = model.decoder_gpi(shared_xb, torch.zeros_like(private_xb))
                recon_shared_y = model.decoder_stn(shared_yb, torch.zeros_like(private_yb))
                loss_shared_recon = F.mse_loss(recon_shared_x, xb) + F.mse_loss(recon_shared_y, yb)

                loss_align_mse = MSE_loss(shared_xb, shared_yb)
                loss_align_cos = cos_sim_loss(shared_xb, shared_yb)
                loss_align = 0.7 * loss_align_mse + 0.3 * loss_align_cos
                loss_orth = orthogonality_loss(shared_xb, private_xb) + orthogonality_loss(shared_yb, private_yb)
                #val_loss += (loss_recon + 0.4 * loss_align + 0.01 * loss_orth).item()
                val_loss += (loss_recon + 0.4 * loss_align + 0.01 * loss_orth + 0*loss_shared_recon).item()
                

                # val_loss += (loss_recon + 0.3 * loss_align).item() #default was loss_recon + 0.1 * loss_align
                val_recon_loss += loss_recon.item()
                val_align_loss += loss_align.item()
                val_orth_loss += loss_orth.item()

                # 🔥 Cosine similarity (mean over B & T)
                cos_sim = F.cosine_similarity(
                    F.normalize(shared_xb, dim=-1),
                    F.normalize(shared_yb, dim=-1),
                    dim=-1
                )
                val_cos_sim += cos_sim.mean().item()

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
        writer.add_scalar("Loss/val_total", avg_val_loss, epoch)
        writer.add_scalar("Loss/val_recon", avg_val_recon, epoch)
        writer.add_scalar("Loss/val_align", avg_val_align, epoch)
        writer.add_scalar("CosineSimilarity/train", avg_cos_sim, epoch)
        writer.add_scalar("CosineSimilarity/val", avg_val_cos_sim, epoch)
        writer.add_scalar("LR", current_lr, epoch)
        writer.add_scalar("Loss/train_orth", avg_orth, epoch)
        writer.add_scalar("Loss/val_orth", avg_val_orth, epoch)

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

        # --- Early Stopping & Checkpoint ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"✅ New best model saved at epoch {epoch+1}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"⏹️ Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

    # Load the best model before returning
    model.load_state_dict(torch.load(model_save_path))
    # torch.save(model, model_save_path)
    # torch.save(val_set, "val_data.pt")
    torch.save({
        'model': model,
        'val_set': val_set
    }, model_save_path)
    
    return model, val_loader, device

def train_spire_3Region(gpi_tensor, stn_tensor,thal_tensor, shared_dim=32, private_dim=32, hidden_dim=64, dropout_prob=0.3, num_epochs=200, batch_size=8,run_name="test1",patience=20,
    model_save_path="best_model3.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = TensorDataset(gpi_tensor, stn_tensor, thal_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1)

    model = SPIRE_3Region(
        input_dim_gpi=gpi_tensor.shape[2],#number of channels
        input_dim_stn=stn_tensor.shape[2],
        input_dim_thal=thal_tensor.shape[2],
        shared_dim=shared_dim,
        private_dim=private_dim,
        hidden_dim=hidden_dim,dropout_prob=dropout_prob
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    writer = SummaryWriter(log_dir="runs/3regions/" + run_name)
    scaler = GradScaler()#Prevents underflow during backprop

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss, total_recon_loss, total_align_loss, total_orth_loss  = 0, 0, 0, 0

        for xb, yb, zb in train_loader: #gpi, stn, thal
            xb, yb, zb = xb.to(device), yb.to(device), zb.to(device)

            # 🚀 Mixed precision forward + loss, Faster training, lower memory usage
            with autocast():
                recon_xb, recon_yb, recon_zb, shared_gpi, shared_stn,shared_thal, private_gpi, private_stn,private_thal = model(xb, yb, zb)

                loss_recon = reconstruction_loss3(xb, recon_xb, yb, recon_yb, zb, recon_zb)

                #align each pair
                loss_align_mse_1 = MSE_loss(shared_gpi, shared_stn)
                loss_align_mse_2 = MSE_loss(shared_gpi, shared_thal)
                loss_align_mse_3 = MSE_loss(shared_stn, shared_thal)

                loss_align_cos_1 = cos_sim_loss(shared_gpi, shared_stn)
                loss_align_cos_2 = cos_sim_loss(shared_gpi, shared_thal)
                loss_align_cos_3 = cos_sim_loss(shared_stn, shared_thal)

                # Combine MSE and Cosine for each pair
                loss_align_1 = 0.7 * loss_align_mse_1 + 0.3 * loss_align_cos_1
                loss_align_2 = 0.7 * loss_align_mse_2 + 0.3 * loss_align_cos_2
                loss_align_3 = 0.7 * loss_align_mse_3 + 0.3 * loss_align_cos_3

                # Average across all three pairs
                loss_align = (loss_align_1 + loss_align_2 + loss_align_3) / 3

                loss_orth = orthogonality_loss(shared_gpi, private_gpi) + orthogonality_loss(shared_stn, private_stn) + orthogonality_loss(shared_thal, private_thal)

                loss = loss_recon + 0.4 * loss_align + 0.01*loss_orth #default was loss_recon + 0.1 * loss_align, a balance between:Preserving high reconstruction accuracy,Encouraging meaningful shared structure

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            total_loss += loss.item()
            total_recon_loss += loss_recon.item()
            total_align_loss += loss_align.item()
            total_orth_loss += loss_orth.item()

        avg_loss = total_loss / len(train_loader)
        avg_recon = total_recon_loss / len(train_loader)
        avg_align = total_align_loss / len(train_loader)
        avg_orth = total_orth_loss / len(train_loader)


        # --- Validation ---
        model.eval()
        val_loss, val_recon_loss, val_align_loss, val_orth_loss = 0, 0, 0, 0

        with torch.no_grad():
            for xb, yb,zb in val_loader:
                xb, yb,zb = xb.to(device), yb.to(device), zb.to(device)
                recon_xb, recon_yb, recon_zb, shared_gpi, shared_stn,shared_thal, private_gpi, private_stn,private_thal = model(xb, yb, zb)

                loss_recon = reconstruction_loss3(xb, recon_xb, yb, recon_yb, zb, recon_zb)

                #align each pair
                loss_align_mse_1 = MSE_loss(shared_gpi, shared_stn)
                loss_align_mse_2 = MSE_loss(shared_gpi, shared_thal)
                loss_align_mse_3 = MSE_loss(shared_stn, shared_thal)

                loss_align_cos_1 = cos_sim_loss(shared_gpi, shared_stn)
                loss_align_cos_2 = cos_sim_loss(shared_gpi, shared_thal)
                loss_align_cos_3 = cos_sim_loss(shared_stn, shared_thal)

                # Combine MSE and Cosine for each pair
                loss_align_1 = 0.7 * loss_align_mse_1 + 0.3 * loss_align_cos_1
                loss_align_2 = 0.7 * loss_align_mse_2 + 0.3 * loss_align_cos_2
                loss_align_3 = 0.7 * loss_align_mse_3 + 0.3 * loss_align_cos_3

                # Average across all three pairs
                loss_align = (loss_align_1 + loss_align_2 + loss_align_3) / 3
                
                loss_orth = orthogonality_loss(shared_gpi, private_gpi) + orthogonality_loss(shared_stn, private_stn) + orthogonality_loss(shared_thal, private_thal)

                val_loss += (loss_recon + 0.4 * loss_align + 0.01 * loss_orth).item() #default was loss_recon + 0.1 * loss_align
                val_recon_loss += loss_recon.item()
                val_align_loss += loss_align.item()
                val_orth_loss += loss_orth.item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_recon = val_recon_loss / len(val_loader)
        avg_val_align = val_align_loss / len(val_loader)
        avg_val_orth = val_orth_loss / len(val_loader)


        scheduler.step(avg_val_loss)

        current_lr = optimizer.param_groups[0]['lr']

        writer.add_scalar("Loss/train_total", avg_loss, epoch)
        writer.add_scalar("Loss/train_recon", avg_recon, epoch)
        writer.add_scalar("Loss/train_align", avg_align, epoch)
        writer.add_scalar("Loss/val_total", avg_val_loss, epoch)
        writer.add_scalar("Loss/val_recon", avg_val_recon, epoch)
        writer.add_scalar("Loss/val_align", avg_val_align, epoch)
        writer.add_scalar("LR", current_lr, epoch)
        writer.add_scalar("Loss/train_orth", avg_orth, epoch)
        writer.add_scalar("Loss/val_orth", avg_val_orth, epoch)


        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

        # --- Early Stopping & Checkpoint ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"✅ New best model saved at epoch {epoch+1}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"⏹️ Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

    # Load the best model before returning
    model.load_state_dict(torch.load(model_save_path))
    # torch.save(model, model_save_path)
    # torch.save(val_set, "val_data.pt")
    torch.save({
        'model': model,
        'val_set': val_set
    }, model_save_path)
    
    return model, val_loader, device