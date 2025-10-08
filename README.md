
# SPIRE-ICLR2026 (Anonymized Submission)

This repository contains an anonymized implementation of **SPIRE** (Shared-Private Inter-Regional Encoder), a multi-region latent modeling framework for intracranial neural recordings (i.e. local field potentails (LFPs)). SPIRE learns to disentangle shared and region-specific dynamics across brain areas using GRU-based encoders and decoders, guided by reconstruction, alignment, and disentanglemnt losses.

This work has been submitted to ICLR 2026.

---

## Overview

SPIRE is trained on off-stimulation data and tested on various conditions (e.g., DBS ON at 85/185/250 Hz) to assess how stimulation modulates latent dynamics. While the dataset is private, the application of SPIRE on synthetic dataset is provided.

---

## Repository Structure
```bash
spire-ICLR2026/
├── scripts/                   # Scripts used to process full dataset (non-public)
│   ├── train_real_all_subjects.py
│   ├── eval_dimSweep.py
│   ├── evaluate_offstim_all_subjects.py
│   ├── figure_4_latent_visualization.py
│   ├── save_onstim_test_all_subjects.py
│   ├── classification_stim_all_subjects.py
│   └── mmd_appendix.py
│
├── scripts_synthetic/                   # Scripts used to process synthectic dataset
│   ├── train_eval_synth.py
│   ├── figure_A1_synth_data.py
│   ├── figure_2_latent_compare_D1.py
│   └── eval_DLAGfitted.py
│
├── src/
│   ├── data/
│   │   ├── data_loader.py               # Functions for loading, preprocessing and segmentation of real neural data
│   │   └── synth_data.py          # Functions to load and generate synthetic data
│   ├── models/
│   │   ├── spire_model.py               # SPIRE model definitions for 2-region
│   │   └── training.py          # training functions for real and synth data
│   ├── utils/
│   │   ├── data_plotting.py               # optional functions to inspect the raw neural data
│   │   ├── losses.py               # All loss functions
│   │   └── training_utils.py             # functions used in training including weight schedules
│   ├── visualization/
│   │   ├── real_visualizer.py               # functions to be used in scripts for plotting real latents, including umap and time trace
│   │   └── synth_visualizer.py           # functions to be used in scripts for plotting syntehtic data and latents
│   ├── evaluate_real.py         # functions used in analysis and evaluation of real data, including extracting latents, variance, onstim classification and mmd
│   └── evaluate_synth.py                   # functions used in evaluation of synth data, including CCA alignment of latents with ground truth
│
├── requirements.txt
└── README.md

```

---
## Installation

1. Clone the repository:
```bash
git clone https://github.com/SPIRE-Anonym/spire-ICLR2026.git
cd spire-ICLR2026
```
2. Create and activate a virtual environment (optional but recommended), then install dependencies:

```bash
pip install -r requirements.txt
```

---
## Synthetic data Scripts:
The scripts_synthetic/ directory includes full experimental workflows used in the study using synthetic data:

- train_eval_synth.py: generates 3 synthetic datasets and runs different variations of the SPIRE model with 4 different seeds. Then evaluates all these models and generates an excel file, which later can be used to compare the performance with DLAG [1] or compare the perfromance of different variants for ablation.
- eval_DLAGfitted.py: DLAG is fitted on synthetic dataset using MATLAB demo function published by Gokcen et al. [2] on 4 different seeds. in This function the performance is evaluated similar to SPIRE.

---
## Full Pipeline Scripts:
The scripts/ directory includes full experimental workflows used in the study using real neural data:

- train_real_all_subjects.py: trains the 2-region model on all real subjects with dimension sweep

- eval_dimSweep: calculates the variance explained by latents to be used in choosing which dimension combination is best for each subject 

- evaluate_offstim_all_subjects.py: using the best dimension for each subject, quantifies the performance of model on unseen offstim test data using CCA and reconstruction MSE

- save_onstim_test_all_subjects.py: extracts latent trajectories under different stimulation conditions (85/185/250 Hz)

- classification_stim_all_subjects.py: classifies stimulation conditions using random forest models trained on different latent types

- mmd_appendix.py: extra measure showing shift of stimulated conditions with respect to offstim in all latents 


---
## Notes on Data
Due to ethical and privacy constraints, the real dataset cannot be shared. However, the repository includes: Complete model code, Reproducible synthetic demos, Fully modular training, evaluation, and visualization tools

For questions or clarifications, please refer to the paper associated with this anonymized submission.

## References

- **DLAG (Delayed Latents Across Groups)** — Gokcen et al., *Nature Computational Science*, 2022.  
  [1]: [Paper (DOI: 10.1038/s43588-022-00282-5)](https://doi.org/10.1038/s43588-022-00282-5) ·
  [2]: [GitHub (MATLAB implementation)](https://github.com/egokcen/DLAG)

