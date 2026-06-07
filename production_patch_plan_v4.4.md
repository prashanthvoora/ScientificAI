# Production Patch Plan for highk_alignn_train_v4.4.py

This file captures the proposed production-hardening and benchmark-reporting changes that can be applied when needed.

## 1. Benchmark-scale reporting fix
- Label log-space values as diagnostic.
- Label linear-k values as the primary benchmark metric.
- Save both `mae_log_k` / `rmse_log_k` and `mae_linear_k` / `rmse_linear_k`.

## 2. Reproducibility hardening
- Set deterministic seeds for Python, NumPy, and PyTorch.
- Disable CuDNN benchmarking.

## 3. Checkpoint loading safety
- Validate checkpoint keys before loading.
- Warn or fail if required heads are missing.

## 4. Input validation
- Validate required columns and finite numeric targets.
- Fail fast on invalid schema or corrupted data.

## 5. Experiment manifest
- Save seed, config, checkpoint epoch, and target-scale metadata.

## 6. Standardized benchmark table
- Print one canonical linear-k benchmark table.
- Keep log-space values in a diagnostic block.

## 7. Resume and error handling
- Add resume support and clear runtime error reporting.
