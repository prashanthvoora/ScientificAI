# Unified Molecular GNN Benchmark: QM9 + MD17

This benchmark supports **both**:

- **QM9** for molecular property prediction
- **MD17** for molecular energy prediction, with optional force supervision/evaluation

## Models compared
- CGCNN-style baseline using `CGConv`
- SchNet
- DimeNet

## Why two dataset modes?

### QM9 mode
Use this when you want to compare models for **scalar molecular properties** such as:
- HOMO
- LUMO
- HOMO-LUMO gap
- dipole moment
- heat capacity
- atomisation energy

### MD17 mode
Use this when you want to compare models for:
- **molecular energy prediction**
- **force prediction** if `--use-forces` is enabled

## Install

```bash
pip install torch torch-geometric pandas numpy matplotlib scikit-learn
```

## Example runs

### 1) QM9 property benchmark
```bash
python train_compare_molecular_gnns.py \
  --dataset qm9 \
  --dataset-root ./data \
  --output-dir ./outputs_qm9_gap \
  --qm9-target-index 4 \
  --epochs 30 \
  --batch-size 32 \
  --max-samples 5000
```

### 2) MD17 energy-only benchmark
```bash
python train_compare_molecular_gnns.py \
  --dataset md17 \
  --dataset-root ./data \
  --output-dir ./outputs_md17_energy \
  --md17-molecule "revised aspirin" \
  --epochs 30 \
  --batch-size 16 \
  --max-samples 5000
```

### 3) MD17 energy + forces benchmark
```bash
python train_compare_molecular_gnns.py \
  --dataset md17 \
  --dataset-root ./data \
  --output-dir ./outputs_md17_energy_forces \
  --md17-molecule "revised aspirin" \
  --use-forces \
  --force-weight 10.0 \
  --epochs 30 \
  --batch-size 8 \
  --max-samples 3000
```

## Outputs
- `metrics_summary.csv`
- `epoch_metrics.csv`
- model checkpoints
- learning curve plot
- bar plots for the main metrics

## Practical guidance

### If your focus is molecules in general
Start with **QM9** to compare the three model families cheaply and quickly.

### If your focus is molecular dynamics / force fields
Move to **MD17** and enable `--use-forces`.

## Important note
This is a strong **research starter benchmark**, not yet a production drug-discovery pipeline.
You will still need:
- proper hyperparameter sweeps
- repeated runs with multiple seeds
- train/val/test protocols aligned to the target benchmark paper
- possibly newer baselines beyond these three
