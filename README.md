
# SPIRE-NeurIPS2025 (Anonymized Submission)

This repository contains an anonymized implementation of **SPIRE** (Shared-Private Inter-Regional Encoder), a multi-region latent modeling framework for intracranial neural recordings (i.e. local field potentails (LFPs)). SPIRE learns to disentangle shared and region-specific dynamics across brain areas using dual GRU-based encoders and decoders, guided by reconstruction, alignment, and orthogonality losses.

This work has been submitted to NeurIPS 2025.

---

## Overview

SPIRE supports:
- **Two-region decoding** (e.g., GPi → STN)
- **Three-region decoding** (e.g., GPi + VO → STN)

It is trained on off-stimulation data and tested on various conditions (e.g., DBS ON at 85/185/250 Hz) to assess how stimulation modulates latent dynamics. While the dataset is private, **fully functional synthetic demos** are provided.

---

## Repository Structure
```bash
spire-neurips2025/
├── examples/                   # Demos and notebooks
│   ├── demo_run_2region.ipynb
│   └── demo_run_3region.ipynb
│
├── scripts/                   # Scripts used to process full dataset (non-public)
│   ├── train_all_subjects.py
│   ├── train_all_subjects_3region.py
│   ├── evaluate_offstim_all_subjects.py
│   ├── evaluate_offstim_all_subjects_3region.py
│   ├── offstim_umap.py
│   ├── save_onstim_test_all_subjects.py
│   ├── shift_onstim_all_subjects.py
│   ├── onstim_umap_trajectories.py
│   └── classification_stim_all_subjects.py
│
├── src/
│   ├── data/
│   │   └── data_loader.py          # Functions to load and preprocess raw neural data
│   ├── models/
│   │   └── spire_model.py          # SPIRE model definitions for 2-region and 3-region
│   ├── utils/
│   │   ├── losses.py               # All loss functions
│   │   └── plotting.py             # Plotting utilities (PSD, time-series)
│   ├── visualization/
│   │   └── umap_utils.py           # Functions for 3D UMAP plots of latent space
│   ├── analysis.py                 # Functions for analysis of the stimulation data, includeing pointwise distribution and classification
│   ├── evaluate.py                # MSE, R², and other offstim model evaluation functions
│   └── train.py                   # Training loop for SPIRE (2-region & 3-region)
│
├── requirements.txt
└── README.md

```

---
## Installation

1. Clone the repository:
```bash
git clone https://github.com/spire-anon/spire-neurips2025.git
cd spire-neurips2025
```
2. Create and activate a virtual environment (optional but recommended), then install dependencies:

```bash
pip install -r requirements.txt
```

---
## Running the Demos

Two Jupyter notebooks are provided for quick testing:

examples/demo_run_2region.ipynb: trains a two-region SPIRE model on synthetic data and evaluates reconstruction.

examples/demo_run_3region.ipynb: demonstrates the three-region model on synthetic data.

To run either demo:

Open the notebook in Jupyter or VS Code.

Run all cells. No private data is required.

---
## Full Pipeline Scripts:
The scripts/ directory includes full experimental workflows used in the study:

- train_all_subjects.py: trains the 2-region model on all real subjects

- evaluate_offstim_all_subjects.py: evaluates reconstruction on off-stim test sets and exports MSE metrics

- offstim_umap.py: visualizes UMAP embeddings of latent variables on off-stim test data

- save_onstim_test_all_subjects.py: extracts latent trajectories under different stimulation conditions (85/185/250 Hz)

- shift_onstim_all_subjects.py: computes pointwise distribution shifts in latent space between off-stim and stimulated conditions

- onstim_umap_trajectories.py: visualizes latent trajectories (e.g., private/shared) under stimulation vs. off-stim

- classification_stim_all_subjects.py: classifies stimulation conditions using random forest models trained on different latent types

- train_all_subjects_3region.py and evaluate_offstim_all_subjects_3region.py: same as above, extended to three-region models

---
## Notes on Data
Due to ethical and privacy constraints, the real dataset cannot be shared. However, the repository includes: Complete model code, Reproducible synthetic demos, Fully modular training, evaluation, and visualization tools

For questions or clarifications, please refer to the paper associated with this anonymized submission.
