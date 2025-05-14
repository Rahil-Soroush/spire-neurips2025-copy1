#align and reconstruction losses
import torch
import torch.nn.functional as F

def reconstruction_loss(x_gpi, recon_gpi, x_stn, recon_stn):
    loss_gpi = F.mse_loss(recon_gpi, x_gpi)
    loss_stn = F.mse_loss(recon_stn, x_stn)
    return loss_gpi + loss_stn

def cos_sim_loss(shared_gpi, shared_stn):
    # Encourage shared latents to be similar (cosine similarity)
    sim = F.cosine_similarity(shared_gpi, shared_stn, dim=-1)  # (B, T)
    return 1 - sim.mean()

def MSE_loss(shared_gpi,shared_stn):
    loss_align = F.mse_loss(shared_gpi, shared_stn)
    return loss_align

def orthogonality_loss(shared, private):
    #try to force that shared parivate learn different things (uncorrelated)
    # Flatten (B, T, D) → (B*T, D)
    s = shared.reshape(-1, shared.shape[-1])
    p = private.reshape(-1, private.shape[-1])
    # Normalize
    s = F.normalize(s, dim=1)
    p = F.normalize(p, dim=1)
    # Inner product → shape (D_s, D_p)
    prod = torch.matmul(s.T, p)
    return torch.norm(prod, p='fro') / prod.numel() #Frobenius norm of the product matrix

def reconstruction_loss3(x_gpi, recon_gpi, x_stn, recon_stn, x_thal, recon_thal):
    loss_gpi = F.mse_loss(recon_gpi, x_gpi)
    loss_stn = F.mse_loss(recon_stn, x_stn)
    loss_thal = F.mse_loss(recon_thal, x_thal)
    return loss_gpi + loss_stn + loss_thal