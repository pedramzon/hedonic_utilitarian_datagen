

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

