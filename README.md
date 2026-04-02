# Enhanced MD17 Sweep Script

This version adds:

- DimeNet++
- MD17 hyperparameter sweep mode
- one combined CSV for all sweep runs
- automatic `Force MAE vs Training Time` plot

## Main output files in sweep mode

- `md17_sweep_all_runs.csv`
- `md17_sweep_epoch_metrics.csv`
- `force_mae_vs_train_time.png`
- `force_mae_vs_num_parameters.png`

## Example command

```bash
python train_compare_molecular_gnns_sweep.py \
  --dataset md17 \
  --dataset-root ./data \
  --output-dir ./outputs_md17_sweep \
  --md17-molecule "revised aspirin" \
  --use-forces \
  --epochs 200 \
  --patience 9999 \
  --sweep-md17 \
  --models dimenet,dimenetpp \
  --sweep-lr 1e-4,3e-4 \
  --sweep-batch-size 2,4 \
  --sweep-cutoff 4.0,5.0 \
  --sweep-force-weight 1.0,10.0 \
  --sweep-hidden-channels 64,128 \
  --sweep-num-blocks 3,4 \
  --sweep-num-bilinear 4,8 \
  --sweep-num-spherical 4,7 \
  --sweep-num-radial 6,8 \
  --sweep-max-num-neighbors 32,64 \
  --sweep-int-emb-size 32,64 \
  --sweep-basis-emb-size 8,16 \
  --sweep-out-emb-channels 128,256 \
  --max-sweep-runs 64
```

## Important note

The full Cartesian product can explode quickly. Use `--max-sweep-runs` to cap total jobs.
