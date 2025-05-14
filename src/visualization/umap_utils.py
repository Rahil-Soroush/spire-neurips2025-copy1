import torch
import numpy as np
import umap
import pandas as pd

def extract_latents_from_test_set(model, gpi_test, stn_test, device):
    """
    Extract shared and private latents from test tensors for 2-region model.
    
    Args:
        model: trained SPIRE dual autoencoder model
        gpi_test, stn_test: torch tensors of shape (N, T, C)
        device: CUDA or CPU

    Returns:
        shared_gpi_all, shared_stn_all,
        private_gpi_all, private_stn_all: all shape (N, T, D)
    """
    model.eval()
    shared_gpi_all, shared_stn_all = [], []
    private_gpi_all, private_stn_all = [], []

    with torch.no_grad():
        for xb, yb in zip(gpi_test, stn_test):
            xb = xb.unsqueeze(0).to(device)
            yb = yb.unsqueeze(0).to(device)

            _, _, shared_gpi, shared_stn, private_gpi, private_stn = model(xb, yb)

            shared_gpi_all.append(shared_gpi.squeeze(0).cpu())
            shared_stn_all.append(shared_stn.squeeze(0).cpu())
            private_gpi_all.append(private_gpi.squeeze(0).cpu())
            private_stn_all.append(private_stn.squeeze(0).cpu())

    return (
        torch.stack(shared_gpi_all),
        torch.stack(shared_stn_all),
        torch.stack(private_gpi_all),
        torch.stack(private_stn_all),
    )

def flatten_latents(latents):
    #Flattens (N, T, D) → (N×T, D)
    N, T, D = latents.shape
    return latents.reshape(N*T, D)

def subsample_group(X, label, target_n=2000, seed=42):
    #Subsamples and attaches a label
    np.random.seed(seed)
    idx = np.random.choice(len(X), size=min(target_n, len(X)), replace=False)
    return X[idx], [label] * len(idx)

def run_umap_and_label(X, labels):
    """
    Function to run UMAP and make labeled DataFrame
    """
    reducer = umap.UMAP(n_neighbors=20, min_dist=0.1, n_components=3, random_state=42)
    X_umap = reducer.fit_transform(X)
    df = pd.DataFrame(X_umap, columns=["UMAP1", "UMAP2", "UMAP3"])
    df["label"] = labels
    return df