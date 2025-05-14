import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score

def extract_latents_by_condition(model, X_test, Y_test, labels, device, label_map):
    """
    Extracts shared and private latents for GPi and STN from test data, grouped by stimulation condition.

    Returns:
        shared_gpi_dict:    dict of condition -> (N, latent_dim) tensor
        shared_stn_dict:    dict of condition -> (N, latent_dim) tensor
        private_gpi_dict:   dict of condition -> (N, latent_dim) tensor
        private_stn_dict:   dict of condition -> (N, latent_dim) tensor
    """
    model.eval()

    shared_gpi_dict = defaultdict(list)
    shared_stn_dict = defaultdict(list)
    private_gpi_dict = defaultdict(list)
    private_stn_dict = defaultdict(list)

    with torch.no_grad():
        for xb, yb, label in zip(X_test, Y_test, labels):
            xb = xb.unsqueeze(0).to(device)
            yb = yb.unsqueeze(0).to(device)
            label_str = label_map[label.item()]

            _, _, shared_gpi, shared_stn, private_gpi, private_stn = model(xb, yb)

            shared_gpi_dict[label_str].append(shared_gpi.squeeze(0).cpu())
            shared_stn_dict[label_str].append(shared_stn.squeeze(0).cpu())
            private_gpi_dict[label_str].append(private_gpi.squeeze(0).cpu())
            private_stn_dict[label_str].append(private_stn.squeeze(0).cpu())

    # Convert lists to stacked tensors
    for d in [shared_gpi_dict, shared_stn_dict, private_gpi_dict, private_stn_dict]:
        for key in d:
            d[key] = torch.stack(d[key])

    return shared_gpi_dict, shared_stn_dict, private_gpi_dict, private_stn_dict

def calculate_pointwise_distribution_shift_sampled(shared_gpi, shared_stn, private_gpi, private_stn, side, setting, subject_id, max_pairs=500, seed=42):
    """
    Computes and samples pairwise distances (cosine & euclidean) from on-stim timepoints to off-stim timepoints.
    Keeps up to max_pairs for each (latent_type, frequency) pair to reduce output size.
    """
    all_latents = {
        "shared_gpi": shared_gpi,
        "shared_stn": shared_stn,
        "private_gpi": private_gpi,
        "private_stn": private_stn,
    }

    rng = np.random.default_rng(seed)
    rows = []

    for latent_type, latent_dict in all_latents.items():
        if "Off" not in latent_dict:
            continue

        off_np = latent_dict["Off"].reshape(-1, latent_dict["Off"].shape[-1]).cpu().numpy()

        for freq in ["85Hz", "185Hz", "250Hz"]:
            if freq not in latent_dict:
                continue

            stim_np = latent_dict[freq].reshape(-1, latent_dict[freq].shape[-1]).cpu().numpy()

            cosine_matrix = cosine_distances(stim_np, off_np)
            euclidean_matrix = euclidean_distances(stim_np, off_np)

            stim_len, off_len = cosine_matrix.shape
            total_pairs = stim_len * off_len

            # Get random flat indices of pairs
            if total_pairs > max_pairs:
                sampled_indices = rng.choice(total_pairs, size=max_pairs, replace=False)
            else:
                sampled_indices = np.arange(total_pairs)

            stim_indices, off_indices = np.unravel_index(sampled_indices, (stim_len, off_len))

            for i, j in zip(stim_indices, off_indices):
                rows.append({
                    "subject": subject_id,
                    "side": side,
                    "setting": setting,
                    "frequency": freq,
                    "latent_type": latent_type,
                    "stim_index": i,
                    "off_index": j,
                    "cosine_distance": cosine_matrix[i, j],
                    "euclidean_distance": euclidean_matrix[i, j],
                })

    return pd.DataFrame(rows)

def prepare_latents_without_averaging(latent_dict):
    """
    Convert (N, T, D) latent tensors into (N*T, D) feature vectors,
    using each timepoint as a sample.
    """
    data = []
    labels = []
    condition_map = {"Off": 0, "85Hz": 1, "185Hz": 2, "250Hz": 3}

    for condition, tensor in latent_dict.items():
        N, T, D = tensor.shape
        flat = tensor.reshape(N * T, D)  # shape: (N*T, D)
        data.append(flat)
        labels.extend([condition_map[condition]] * (N * T))

    X = np.vstack(data)
    y = np.array(labels)
    return X, y

def calculate_RF_accuracy(shared_gpi, shared_stn, private_gpi, private_stn, side, setting, subject_id):
    latent_sets = {
        "shared_gpi": shared_gpi,
        "shared_stn": shared_stn,
        "private_gpi": private_gpi,
        "private_stn": private_stn,
    }

    results_accuracy = []
    results_importance = []

    for latent_type, latent_dict in latent_sets.items():
        X, y = prepare_latents_without_averaging(latent_dict)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=0.2, random_state=42
        )

        # Normalize
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        # Save classification accuracy
        results_accuracy.append({
            "subject": subject_id,
            "side": side,
            "setting": setting,
            "latent_type": latent_type,
            "accuracy": acc,
            "precision_macro": precision_score(y_test, y_pred, average='macro'),
            "recall_macro": recall_score(y_test, y_pred, average='macro'),
            "f1_macro": f1_score(y_test, y_pred, average='macro'),
            "n_samples": len(y)
        })

        # Save feature importances
        for i, imp in enumerate(clf.feature_importances_):
            results_importance.append({
                "subject": subject_id,
                "side": side,
                "setting": setting,
                "latent_type": latent_type,
                "feature_dim": i,
                "importance": imp,
            })

    return pd.DataFrame(results_accuracy), pd.DataFrame(results_importance)