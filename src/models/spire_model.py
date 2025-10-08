import torch
import torch.nn as nn
import torch.nn.functional as F


# Dual-latent encoder, which provides shared and private dynamics
class LatentEncoder(nn.Module):
    """
    Encodes input signals into shared and private latent representations using a GRU.
    """
    def __init__(self, input_channels, shared_dim=32, private_dim=32, hidden_dim=64, dropout_prob = 0.3):
        super().__init__()
        self.gru = nn.GRU(input_channels, hidden_dim, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(dropout_prob)
        self.shared_proj = nn.Linear(hidden_dim , shared_dim)
        self.private_proj = nn.Linear(hidden_dim , private_dim)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, time_steps, input_dim)
        Returns:
            shared: Shared latent representation (B, T, shared_dim)
            private: Private latent representation (B, T, private_dim)
        """
        out, _ = self.gru(x)  # (B, T, 2*H)
        out = self.dropout(out)  
        shared = self.shared_proj(out)
        private = self.private_proj(out)
        return shared, private

# Decoder
class LatentDecoder(nn.Module):
    """
    Generates reconstructed signals using shared and private latent representations
    """
    def __init__(self, shared_dim, private_dim, output_channels, hidden_dim=64, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(shared_dim + private_dim, hidden_dim, batch_first=True, num_layers=num_layers)
        self.out = nn.Linear(hidden_dim, output_channels)

    def forward(self, shared, private):
        """
        Args:
            shared: Shared latent representation (B, T, shared_dim)
            private: Private latent representation (B, T, private_dim)
        Returns:
            out: reconstructed tensor of shape (batch_size, time_steps, output_channels)
        """
        x = torch.cat([shared, private], dim=-1)
        out, _ = self.gru(x)
        return self.out(out)

class ConvAlign1D(nn.Module):
    """
    depthwise temporal aligner initialized as an impulse
    """
    def __init__(self, dim: int, kernel_size: int = 9):
        super().__init__()
        assert kernel_size % 2 == 1, "use odd kernel_size"
        self.K = kernel_size
        self.center = kernel_size // 2
        # depthwise conv over time, one filter per latent dim
        self.conv = nn.Conv1d(
            in_channels=dim, out_channels=dim,
            kernel_size=kernel_size, padding=self.center,
            groups=dim, bias=False
        )
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        # make every per-dim filter an impulse: [0,...,0,1,0,...,0]
        w = torch.zeros_like(self.conv.weight)   # [dim, 1, K]
        w[:, 0, self.center] = 1.0
        self.conv.weight.copy_(w)

    def forward(self, z_bt_d: torch.Tensor) -> torch.Tensor:
        # z: [B, T, D] -> conv1d expects [B, D, T]
        z_bdt = z_bt_d.permute(0, 2, 1)
        y_bdt = self.conv(z_bdt)
        return y_bdt.permute(0, 2, 1)            # back to [B, T, D]

    # small regularizer: keep filters near impulses and sum≈1
    def reg_loss(self, strength_delta: float = 1.0, strength_sum: float = 0.1):
        w = self.conv.weight.squeeze(1)          # [D, K]
        delta = torch.zeros_like(w)
        delta[:, self.center] = 1.0
        loss_delta = F.mse_loss(w, delta)        # impulse prior
        loss_sum   = F.mse_loss(w.sum(dim=1), torch.ones(w.size(0), device=w.device))
        return strength_delta * loss_delta + strength_sum * loss_sum

# dual autoencoder
class SPIREAutoencoder(nn.Module):
    """
    A dual autoencoder model using the encoder and decoder defined
    """
    def __init__(self, input_dim_gpi, input_dim_stn, shared_dim=32, private_dim=32, hidden_dim=64, dropout_prob=0.3):
        super().__init__()

        self.use_identity_align = False # for ablation control

        self.encoder_gpi = LatentEncoder(input_dim_gpi, shared_dim, private_dim, hidden_dim, dropout_prob)
        self.encoder_stn = LatentEncoder(input_dim_stn, shared_dim, private_dim, hidden_dim, dropout_prob)

        self.decoder_gpi = LatentDecoder(shared_dim, private_dim, input_dim_gpi, hidden_dim,num_layers = 2)
        self.decoder_stn = LatentDecoder(shared_dim, private_dim, input_dim_stn, hidden_dim, num_layers =2)

        # kernel_size=9 (±4 samples) is enough for your tvd_amplitude=3. If you might use larger jitter, bump to 13
        self.align_x2y = ConvAlign1D(shared_dim, kernel_size=9)      # x->y
        self.align_y2x = ConvAlign1D(shared_dim, kernel_size=9)      # y->x

        # NEW: light linear maps between shared spaces (init as Identity)
        self.map_y2x = nn.Linear(shared_dim, shared_dim, bias=False)
        self.map_x2y = nn.Linear(shared_dim, shared_dim, bias=False)
        with torch.no_grad():
            self.map_y2x.weight.copy_(torch.eye(shared_dim))
            self.map_x2y.weight.copy_(torch.eye(shared_dim))

    def align_y_to_x(self, z_y_bt_d):
        # conv-align in time, then map dims if needed
        if self.use_identity_align: #for ablation
            return self.map_y2x(z_y_bt_d)  # mapper only (init=I)
        return self.map_y2x(self.align_y2x(z_y_bt_d))

    def align_x_to_y(self, z_x_bt_d):
        if self.use_identity_align: #for ablation
            return self.map_x2y(z_x_bt_d)
        return self.map_x2y(self.align_x2y(z_x_bt_d))

    def align_regularizer(self):
        # expose combined reg for the training loop
        return self.align_x2y.reg_loss() + self.align_y2x.reg_loss()

    # small identity penalty for the mappers (use early, then decay)
    def mapper_identity_loss(self):
        I = torch.eye(self.map_y2x.weight.size(0), device=self.map_y2x.weight.device)
        return (self.map_y2x.weight - I).pow(2).mean() + (self.map_x2y.weight - I).pow(2).mean()

    def forward(self, x_gpi, x_stn, private_gate: float = 1.0):
        """
        Args:
            signals from two regions (B, T, C), C can be different for the two regions
            private_gate (αp): 0.0 during warm-start (shared-only),
                           linearly ramped to 1.0 over a few epochs.
        Returns:
            reconstructed signals (B, T, C), shared latents of each region (B, T, shared_dim),
              private latents of each region(B, T, private_dim)
        """
        shared_gpi, private_gpi = self.encoder_gpi(x_gpi)
        shared_stn, private_stn = self.encoder_stn(x_stn)

        # <<< gate private latents here >>>
        if private_gate != 1.0:
            private_gpi = private_gate * private_gpi
            private_stn = private_gate * private_stn

        recon_gpi = self.decoder_gpi(shared_gpi, private_gpi)
        recon_stn = self.decoder_stn(shared_stn, private_stn)

        return recon_gpi, recon_stn, shared_gpi, shared_stn, private_gpi, private_stn
    
