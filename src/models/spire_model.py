import torch
import torch.nn as nn


# Dual-latent encoder, which provides shared and private dynamics
class LatentEncoder(nn.Module):
    """
    Encodes input signals into shared and private latent representations using a bidirectional GRU.
    """
    def __init__(self, input_channels, shared_dim=32, private_dim=32, hidden_dim=64, dropout_prob = 0.3):
        super().__init__()
        self.gru = nn.GRU(input_channels, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.shared_proj = nn.Linear(hidden_dim * 2, shared_dim)
        self.private_proj = nn.Linear(hidden_dim * 2, private_dim)

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

# dual autoencoder
class SPIRE(nn.Module):
    """
    A dual autoencoder model using the encoder and decoder defined
    """
    def __init__(self, input_dim_gpi, input_dim_stn, shared_dim=32, private_dim=32, hidden_dim=64, dropout_prob=0.3):
        super().__init__()
        self.encoder_gpi = LatentEncoder(input_dim_gpi, shared_dim, private_dim, hidden_dim, dropout_prob)
        self.encoder_stn = LatentEncoder(input_dim_stn, shared_dim, private_dim, hidden_dim, dropout_prob)

        self.decoder_gpi = LatentDecoder(shared_dim, private_dim, input_dim_gpi, hidden_dim,num_layers = 2)
        self.decoder_stn = LatentDecoder(shared_dim, private_dim, input_dim_stn, hidden_dim, num_layers =2)

    def forward(self, x_gpi, x_stn):
        """
        Args:
            signals from two regions (B, T, C), C can be different for the two regions
        Returns:
            reconstructed signals (B, T, C), shared latents of each region (B, T, shared_dim),
              private latents of each region(B, T, private_dim)
        """
        shared_gpi, private_gpi = self.encoder_gpi(x_gpi)
        shared_stn, private_stn = self.encoder_stn(x_stn)

        recon_gpi = self.decoder_gpi(shared_gpi, private_gpi)
        recon_stn = self.decoder_stn(shared_stn, private_stn)

        return recon_gpi, recon_stn, shared_gpi, shared_stn, private_gpi, private_stn
    
#thriple AE
class SPIRE_3Region(nn.Module):
    def __init__(self, input_dim_gpi, input_dim_stn, input_dim_thal, shared_dim=32, private_dim=32, hidden_dim=64, dropout_prob=0.3):
        super().__init__()
        self.encoder_gpi = LatentEncoder(input_dim_gpi, shared_dim, private_dim, hidden_dim, dropout_prob)
        self.encoder_stn = LatentEncoder(input_dim_stn, shared_dim, private_dim, hidden_dim, dropout_prob)
        self.encoder_thal = LatentEncoder(input_dim_thal, shared_dim, private_dim, hidden_dim, dropout_prob)

        self.decoder_gpi = LatentDecoder(shared_dim, private_dim, input_dim_gpi, hidden_dim, num_layers = 2)
        self.decoder_stn = LatentDecoder(shared_dim, private_dim, input_dim_stn, hidden_dim, num_layers = 2)
        self.decoder_thal = LatentDecoder(shared_dim, private_dim, input_dim_thal, hidden_dim, num_layers = 2)

    def forward(self, x_gpi, x_stn, x_thal):
        shared_gpi, private_gpi = self.encoder_gpi(x_gpi)
        shared_stn, private_stn = self.encoder_stn(x_stn)
        shared_thal, private_thal = self.encoder_thal(x_thal)

        recon_gpi = self.decoder_gpi(shared_gpi, private_gpi)
        recon_stn = self.decoder_stn(shared_stn, private_stn)
        recon_thal = self.decoder_thal(shared_thal, private_thal)

        return recon_gpi, recon_stn,recon_thal, shared_gpi, shared_stn,shared_thal, private_gpi, private_stn,private_thal