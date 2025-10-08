#align and reconstruction losses
import torch
import torch.nn.functional as F

def reconstruction_loss(x_gpi, recon_gpi, x_stn, recon_stn):
    loss_gpi = F.mse_loss(recon_gpi, x_gpi)
    loss_stn = F.mse_loss(recon_stn, x_stn)
    return loss_gpi + loss_stn

def orthogonality_loss(shared, private, eps=1e-6):
    # shared, private: (B, T, D)
    S = shared.reshape(-1, shared.size(-1))
    P = private.reshape(-1, private.size(-1))
    # standardize per feature
    S = (S - S.mean(0)) / (S.std(0) + eps)
    P = (P - P.mean(0)) / (P.std(0) + eps)
    N = S.size(0)
    # C = (S.T @ P) / (N - 1)          # <-- normalize by N-1
    # guard small N and symmetrize the cross-cov (not strictly needed, but stable)
    C = (S.T @ P) / max(1, N - 1)
    return (C**2).mean()

def vicreg_loss(z1, z2, eps=1e-4): #robust to scale and phase jitter
    # flatten time+batch, center
    z1 = z1.reshape(-1, z1.size(-1)); z2 = z2.reshape(-1, z2.size(-1))
    z1 = z1 - z1.mean(0, keepdim=True); z2 = z2 - z2.mean(0, keepdim=True)
    inv = F.mse_loss(z1, z2)

    def var_term(z):
        s = z.std(0) + eps
        return torch.mean(F.relu(1.0 - s))  # keep std >= 1

    def cov_term(z):
        N = z.size(0)
        # C = (z.T @ z) / (N - 1)
        # guard small N and symmetrize
        C = (z.T @ z) / max(1, N - 1)
        C = 0.5 * (C + C.T)
        off = C - torch.diag(torch.diag(C))
        return (off**2).sum() / z.size(1)

    return inv + var_term(z1) + var_term(z2) + cov_term(z1) + cov_term(z2)



