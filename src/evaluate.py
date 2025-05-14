#basic reconstruction and loss evaluation
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import pandas as pd


def evaluate_spire(model, val_loader, device):
    """
    for one validation sample calculates MSE and r2 and plots input and recostructed STN 
    """
    model.eval()
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            recon_xb, recon_yb, shared_xb, shared_yb, _ ,_ = model(xb, yb)
            break  # visualize just one batch

    mse = F.mse_loss(recon_yb, yb).item()
    r2 = 1 - torch.sum((yb - recon_yb)**2) / torch.sum((yb - torch.mean(yb))**2)

    plt.figure(figsize=(14, 4))
    plt.plot(yb.cpu().flatten(), label='True STN')
    plt.plot(recon_yb.cpu().flatten(), label='Reconstructed STN', linestyle='--')
    plt.title("SPIRE Model: True vs Reconstructed STN")
    plt.xlabel("Time (flattened)")
    plt.ylabel("Normalized Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"📊 Validation MSE: {mse:.4f}")
    print(f"📈 Validation R² Score: {r2:.4f}")

def evaluate_spire_3Region(model, val_loader, device):
    """
    for the three region model
    for one validation sample calculates MSE and r2 and plots input and recostructed STN 
    """
    model.eval()
    with torch.no_grad():
        for xb, yb, zb in val_loader:
            xb, yb,zb = xb.to(device), yb.to(device),zb.to(device)
            recon_xb, recon_yb, recon_zb, shared_xb, shared_yb,shared_zb, _ ,_, _ = model(xb, yb, zb)
            break  # visualize just one batch

    mse = F.mse_loss(recon_yb, yb).item()
    r2 = 1 - torch.sum((yb - recon_yb)**2) / torch.sum((yb - torch.mean(yb))**2)

    plt.figure(figsize=(14, 4))
    plt.plot(yb.cpu().flatten(), label='True STN')
    plt.plot(recon_yb.cpu().flatten(), label='Reconstructed STN', linestyle='--')
    plt.title("SPIRE 3 region Model: True vs Reconstructed STN")
    plt.xlabel("Time (flattened)")
    plt.ylabel("Normalized Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"📊 Validation MSE: {mse:.4f}")
    print(f"📈 Validation R² Score: {r2:.4f}")

def get_mse_r2_df_per_sample(model, gpi_test, stn_test, device,side, subject_id=None):
    model.eval()

    records = []

    for i in range(gpi_test.shape[0]):
        xb = gpi_test[i:i+1].to(device)  # shape: (1, T, C)
        yb = stn_test[i:i+1].to(device)

        with torch.no_grad():
            recon_xb, recon_yb, shared_xb, shared_yb, private_xb, private_yb = model(xb, yb)

            # Decode variants
            outputs = {
                "gpi_full": recon_xb,
                "gpi_private": model.decoder_gpi(torch.zeros_like(shared_xb), private_xb),
                "gpi_shared_gpi": model.decoder_gpi(shared_xb, torch.zeros_like(private_xb)),
                "gpi_shared_stn": model.decoder_gpi(shared_yb, torch.zeros_like(private_xb)),
                "stn_full": recon_yb,
                "stn_private": model.decoder_stn(torch.zeros_like(shared_yb), private_yb),
                "stn_shared_stn": model.decoder_stn(shared_yb, torch.zeros_like(private_yb)),
                "stn_shared_gpi": model.decoder_stn(shared_xb, torch.zeros_like(private_yb)),
            }

            targets = {
                "gpi": xb,
                "stn": yb,
            }

            # For each condition, calculate MSE and R²
            for key, output in outputs.items():
                region = key.split("_")[0]
                target = targets[region]
                mse = F.mse_loss(output, target, reduction='mean').item()
                # R²: flatten (1, T, C) → (T*C,)
                r2 = r2_score(target.cpu().numpy().flatten(), output.cpu().numpy().flatten())
                records.append({
                    "subject": subject_id,
                    'side': side,
                    "sample": i,
                    "condition": key,
                    "mse": mse,
                    "r2": r2
                })

    return pd.DataFrame(records)

def get_mse_r2_df_per_sample_3Regions(model, gpi_test, stn_test, thal_test,device,side, subject_id=None):
    model.eval()

    records = []

    for i in range(gpi_test.shape[0]):
        xb = gpi_test[i:i+1].to(device)  # shape: (1, T, C)
        yb = stn_test[i:i+1].to(device)
        zb = thal_test[i:i+1].to(device)

        with torch.no_grad():
            recon_xb, recon_yb,recon_zb, shared_xb, shared_yb,shared_zb, private_xb, private_yb, private_zb = model(xb, yb,zb)

            # Decode variants
            outputs = {
                "gpi_full": recon_xb,
                "gpi_private": model.decoder_gpi(torch.zeros_like(shared_xb), private_xb),
                "gpi_shared_gpi": model.decoder_gpi(shared_xb, torch.zeros_like(private_xb)),
                "gpi_shared_stn": model.decoder_gpi(shared_yb, torch.zeros_like(private_xb)),
                "gpi_shared_thal": model.decoder_gpi(shared_zb, torch.zeros_like(private_xb)),
                "stn_full": recon_yb,
                "stn_private": model.decoder_stn(torch.zeros_like(shared_yb), private_yb),
                "stn_shared_stn": model.decoder_stn(shared_yb, torch.zeros_like(private_yb)),
                "stn_shared_gpi": model.decoder_stn(shared_xb, torch.zeros_like(private_yb)),
                "stn_shared_thal": model.decoder_stn(shared_zb, torch.zeros_like(private_yb)),
                "thal_full": recon_zb,
                "thal_private": model.decoder_thal(torch.zeros_like(shared_zb), private_zb),
                "thal_shared_thal": model.decoder_thal(shared_zb, torch.zeros_like(private_zb)),
                "thal_shared_gpi": model.decoder_thal(shared_xb, torch.zeros_like(private_zb)),
                "thal_shared_stn": model.decoder_thal(shared_yb, torch.zeros_like(private_zb)),
            }

            targets = {
                "gpi": xb,
                "stn": yb,
                "thal": zb,
            }

            # For each condition, calculate MSE and R²
            for key, output in outputs.items():
                region = key.split("_")[0]
                target = targets[region]
                mse = F.mse_loss(output, target, reduction='mean').item()
                # R²: flatten (1, T, C) → (T*C,)
                r2 = r2_score(target.cpu().numpy().flatten(), output.cpu().numpy().flatten())
                records.append({
                    "subject": subject_id,
                    'side': side,
                    "sample": i,
                    "condition": key,
                    "mse": mse,
                    "r2": r2
                })

    return pd.DataFrame(records)