# Reproducibility package: AS-GNN synthetic neuromarketing dataset & PPN training

This package addresses the reviewer's reproducibility comment by (1) fully
documenting, in runnable code, how the synthetic multimodal streams and
ground-truth hedonic/utilitarian labels were generated, (2) providing complete
training details and code for the Pruning Policy Network (PPN), and (3)
supplying a representative public data sample.

## Contents

- `generate_synthetic_dataset.py` — generates the full 12,000-window
  (400 sessions x 30 windows) synthetic dataset described in the paper's
  Results section: latent h,u ~ Beta(2,2) per window; per-modality raw
  signal simulation (64-ch/256 Hz EEG, 500 Hz eye-tracking, 25 Hz GSR,
  scalar LLM-sentiment); derived aggregate features
  (`frontal_eeg_bandpower`, `parietal_eeg_bandpower`, `fixation_duration`,
  `gsr_level`, `llm_sentiment`); joint response `r = 0.5*h + 0.5*u` (Eq. 4);
  and the thresholded discrete label (hedonic-dominant if `r >= 0.5`).
- `train_ppn.py` — trains the modality-specific Pruning Policy Network by
  REINFORCE with a moving-average baseline, using every hyperparameter
  stated in the paper (Gaussian policy, sigma_explore = 0.05; reward
  weights lambda_A = 0.7, lambda_S = 0.3; Adam, lr = 1e-4; discount
  gamma_RL = 0.95 over 30-window episodes; batch size 64 episodes/update;
  baseline decay 0.9; 200 updates = 12,800 episodes).
- `sample_dataset.csv` — a representative public sample: 20 sessions
  (600 windows) generated with seed 0.
- `sample_dataset_raw_sample.npz` — raw EEG (64 x 1280), gaze (2 x 2500),
  and GSR (125,) arrays for the first 5 windows of that sample, so the
  raw-signal generation logic (not just the aggregate features) is
  independently checkable.
- `sample_dataset_meta.json` / `dataset_seed0_meta.json` — machine-readable
  record of every generation constant (sampling rates, noise sigma, Beta
  parameters, lambda, label threshold, seed) for full traceability.
- `ppn_history_full.json` — training-reward log from a full 200-update run,
  included so the plateau behavior mentioned in the paper can be inspected
  without re-running training.

## How to regenerate everything

```bash
# Full 12,000-window dataset, one seed (repeat with --seed 1..4 for the
# paper's N=5 seeds/runs protocol)
python generate_synthetic_dataset.py --seed 0 --n_sessions 400 \
    --windows_per_session 30 --out dataset_seed0.csv

# Small public sample with raw signal arrays attached
python generate_synthetic_dataset.py --seed 0 --n_sessions 20 \
    --windows_per_session 30 --out sample_dataset.csv --raw_sample 5

# Train the Pruning Policy Network (paper protocol: 200 updates)
python train_ppn.py --seed 0 --n_updates 200
```

Both scripts depend only on `numpy` and `pandas`; no GPU or deep-learning
framework is required to reproduce the data-generation or PPN-training
logic described here (the full AS-GNN classifier itself, which is not
included in this package, was implemented in PyTorch per the paper).

## Points requiring author confirmation before deposit

These are flagged transparently in code comments so nothing is silently
guessed on your behalf:

1. **Exact nonlinearity for "monotonically increasing function of h/u".**
   The paper states each feature is such a function with additive
   Gaussian noise (sigma = 0.15) but does not name the specific function.
   This code uses the identity mapping (feature = latent + noise) as the
   simplest function consistent with every explicit statement in the
   paper. If the original experiments used a different nonlinearity
   (e.g., a sigmoid or power-law warp of h/u), only the `_shape()`
   helper in `generate_synthetic_dataset.py` needs to change.
2. **PPN modality count K used to obtain "145 total parameters."**
   With input dim `3+K`, hidden 32, the parameter count is
   `32*(3+K) + 65`. This does not resolve to 145 for any integer K
   (K=5, as used here, gives 321). Please confirm the exact modality
   grouping/K used originally so the architecture in `train_ppn.py`
   can be adjusted to match the reported parameter count exactly.
3. **`classifier_reward_fn` in `train_ppn.py` is a documented placeholder**
   standing in for real AS-GNN validation accuracy/sparsity feedback,
   since the classifier itself is a separate, larger PyTorch codebase
   not reproduced here. It is built to reproduce the *qualitative*
   Table 3 finding (more edges retained for the higher-sampling-rate
   modality) but is not the literal reward signal used to produce the
   paper's headline F1/latency numbers. This should be made explicit if
   this script is cited as producing Table 2/3's exact figures, versus
   as a faithful, runnable specification of the training *procedure*.

## Suggested replacement Data Availability statement

> The code used to generate the synthetic multimodal neuromarketing
> dataset and to train the Pruning Policy Network, together with a
> representative data sample (600 windows, including raw EEG,
> eye-tracking, and GSR signal arrays for a subset of windows), is
> publicly available at [REPOSITORY URL / DOI TO BE INSERTED]. The full
> 12,000-window dataset used in the reported experiments can be
> regenerated exactly from this code using the documented random seeds
> (0-4). The AS-GNN classifier implementation is available at
> [REPOSITORY URL / DOI TO BE INSERTED, or: available from the
> corresponding author on reasonable request, if not yet public].

Replace the bracketed placeholders with your actual repository (e.g., a
Zenodo-archived GitHub/OSF repo, which gives you a citable DOI) once you
upload this package.
