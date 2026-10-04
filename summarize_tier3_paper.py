#!/usr/bin/env python3
"""Summarize local Tier-3 seed runs without exporting records or checkpoints.
Stdlib only. Prefer exact v4.60.4 JSON reports; fall back to rounded console logs.
Usage: python summarize_tier3_paper.py --runs highk_project/reruns
       python summarize_tier3_paper.py --logs seed42.log seed124.log seed777.log
"""
import argparse
import csv
import glob
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
METRICS = ['N', 'ln_MAE', 'ln_RMSE', 'linear_MAE', 'linear_RMSE',
           'linear_MAD', 'linear_MAD_to_MAE']
CONFIG = ['epochs', 'min_epochs', 'patience', 'batch_size', 'scheduler',
          'max_grad_norm', 'process_delta_bound', 'process_delta_use_categorical',
          'tier3_use_structural_adapter', 'tier3_structural_adapter_rank',
          'tier3_structural_adapter_learning_rate', 'tier3_process_learning_rate',
          'tier3_structural_adapter_weight_decay', 'tier3_process_weight_decay',
          'tier3_adapter_magnitude_weight', 'tier3_adapter_center_weight',
          'tier3_process_center_weight', 'tier3_adapter_process_orthogonal_weight']


def read_json(path, default=None):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f'Invalid report {path}: {error}') from error


def last(text, pattern, cast=float):
    matches = re.findall(pattern, text, flags=re.I)
    return cast(matches[-1]) if matches else None


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def collect(root=None, logfile=None):
    audit_dir = root / 'reports/tier3_preprocessing' if root else None
    if logfile is None and root:
        logfile = root / 'train.log'
    text = logfile.read_text(errors='replace') if logfile and logfile.exists() else ''
    evaluation_log = root / 'evaluate.log' if root else None
    eval_text = evaluation_log.read_text(errors='replace') if evaluation_log and evaluation_log.exists() else ''
    load = lambda name, default=None: read_json(audit_dir / name, default) if audit_dir else default
    result = load('results_training.json') or load('results_evaluation.json')
    audit = load('preprocessing_audit.json', {})
    parity = load('evaluation_parity.json', {})
    runtime = load('runtime.json', {})
    config = load('effective_config.json', {})
    label = str(root or logfile)
    path_seed = last(label, r'seed[_-]?(\d+)', int)
    logged_seeds = set(re.findall(r'T3-PROC-RESET:.*?seed=(\d+)', text))
    if len(logged_seeds) > 1:
        raise ValueError(f'{label}: multiple seeds in one log; use separate per-run logs')
    seed = result.get('seed') if result else audit.get('training_seed')
    if seed is None:
        seed = int(next(iter(logged_seeds))) if logged_seeds else path_seed
    if seed is None:
        raise ValueError(f'{label}: seed unavailable; use a filename containing seed42, seed124, etc.')
    if path_seed is not None and int(seed) != path_seed:
        raise ValueError(f'{label}: filename seed disagrees with report/log seed')
    if logged_seeds and int(seed) != int(next(iter(logged_seeds))):
        raise ValueError(f'{label}: report seed disagrees with log seed')
    warnings = []
    counts = audit.get('split_counts', {})
    if not counts:
        match = re.findall(r'PREPROCESSING AUDIT PASS: TRAIN=(\d+) VAL=(\d+) TEST=(\d+)', text)
        if match:
            counts = dict(zip(['TRAIN', 'VAL', 'TEST'], map(int, match[-1])))
    entry = dict(seed=int(seed), precision='exact JSON' if result else 'rounded log',
        selected_epoch=result.get('selected_epoch') if result else last(text, r'best_epoch=(\d+)', int),
        selected_VAL_ln_MAE=result.get('selected_VAL_ln_MAE') if result else
            last(text, r'Training complete\. Best val_MAE=('+NUMBER+r')'),
        stopping_epoch=runtime.get('stopping_epoch') or last(text, r'Epoch\s+(\d+)/\d+\s+loss=', int),
        split_counts=counts, full_metrics={}, comparator_metrics={}, paired_ln_MAE={},
        table_IV={}, preprocessing={}, checks={}, runtime=runtime, warnings=warnings)
    if result:
        entry['full_metrics'] = result.get('metrics', {}).get('full', {})
        entry['comparator_metrics'] = {k: v for k, v in result.get('metrics', {}).items() if k != 'full'}
        entry['paired_ln_MAE'] = result.get('paired_ln_MAE', {})
        config = config or result.get('configuration', {})
    else:
        patterns = {
            'ln_MAE': r'Diagnostic log-space\s+MAE\s*=\s*('+NUMBER+r')',
            'ln_RMSE': r'Diagnostic log-space\s+RMSE\s*=\s*('+NUMBER+r')',
            'linear_MAE': r'PRIMARY \(linear-k exact\)\s+MAE\s*=\s*('+NUMBER+r')',
            'linear_RMSE': r'PRIMARY \(linear-k exact\)\s+RMSE\s*=\s*('+NUMBER+r')',
            'linear_MAD': r'PRIMARY \(linear-k exact\)\s+MAD\s*=\s*('+NUMBER+r')',
            'linear_MAD_to_MAE': r'PRIMARY \(linear-k exact\).*?MAD:MAE\s*=\s*('+NUMBER+r')',
            'N': r'PRIMARY \(linear-k exact\).*?\(N=(\d+)\)',
        }
        entry['full_metrics'] = {key: last(text, pattern, int if key == 'N' else float)
                                 for key, pattern in patterns.items()}
        best = re.findall(r'BEST TEST\s*:\s*log_MAE=('+NUMBER+r')\s+linear_MAE=('+NUMBER+r')\s+MAD:MAE=('+NUMBER+r')\s+best_epoch=(\d+)', text)
        if best:
            a, b, c, epoch = best[-1]
            entry['full_metrics'].update(ln_MAE=float(a), linear_MAE=float(b), linear_MAD_to_MAE=float(c))
            entry['selected_epoch'] = int(epoch)
        warnings.append('Log-only metrics are rounded; paired CIs/configuration may be unavailable.')
    entry['table_IV'] = {key: config[key] for key in CONFIG if key in config}
    entry['table_IV']['selection'] = 'VAL ln-MAE, including epoch zero' if audit else None
    entry['preprocessing'] = {key: audit[key] for key in ['schema', 'statistics_fit_partition',
        'numeric_fields', 'log_fields', 'categorical_vocabulary', 'alias_rules', 'categorical_indices',
        'mask_semantics', 'missing_imputation', 'scale_policy', 'process_input_dim',
        'TRAIN_source_counts', 'split_sha256', 'prior_sha256'] if key in audit}
    # Compare fitting parameters between seeds without exporting numerical input values.
    if audit:
        encoded = json.dumps({k: audit.get(k) for k in ['num_mean', 'num_std',
            'categorical_vocabulary', 'alias_rules']}, sort_keys=True).encode()
        entry['preprocessing']['encoding_sha256'] = hashlib.sha256(encoded).hexdigest()
    history = read_json(root / 'reports/tier3_training_history_normal.json', []) if root else []
    drifts = [row.get('frozen_base_max_abs_change') for row in history]
    drifts = [value for value in drifts if finite(value)]
    if not drifts:
        drifts = [float(v) for v in re.findall(r'frozen_max=('+NUMBER+r')', text)]
    entry['checks'] = {
        'preprocessing_audit_logged': 'PREPROCESSING AUDIT PASS' in text if text else None,
        'TRAIN_only_fit_recorded': audit.get('statistics_fit_partition') == 'TRAIN' if audit else None,
        'input_dim': audit.get('process_input_dim'),
        'evaluation_parity_PASS': parity.get('PASS'),
        'evaluation_parity_N': parity.get('N'),
        'evaluation_max_abs_difference': parity.get('max_abs_difference'),
        'max_frozen_base_change': max(drifts) if drifts else None,
        'training_export_logged': 'EVIDENCE EXPORT PASS: training N=30' in text if text else None,
        'traceback_in_logs': 'Traceback (most recent call last)' in text+eval_text,
    }
    if not result and 'EVIDENCE EXPORT PASS: training' not in text and 'BEST TEST' not in text:
        warnings.append('No completed selected-checkpoint TEST export found.')
    if entry['full_metrics'].get('N') is None:
        warnings.append('TEST metric population count is unavailable.')
    if parity.get('PASS') is not True:
        warnings.append('Standalone evaluation parity has not been verified in the saved report.')
    if entry['checks']['traceback_in_logs']:
        warnings.append('A traceback appears in the logs; inspect the run before publication use.')
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--runs', nargs='+', help='Run directories or parent directory containing seed runs')
    group.add_argument('--logs', nargs='+', help='Per-seed log files (wildcards accepted)')
    parser.add_argument('--out', default='tier3_paper_summary', help='Output filename prefix')
    args = parser.parse_args()
    sources = []
    for value in args.runs or args.logs:
        matches = sorted(glob.glob(value))
        if not matches:
            parser.error(f'No matching path: {value}')
        for match in matches:
            path = Path(match)
            if args.logs:
                if not path.is_file():
                    parser.error(f'Not a log file: {path}')
                root = path.parent if (path.parent / 'reports/tier3_preprocessing').is_dir() else None
                sources.append((root, path))
            elif (path / 'reports').is_dir():
                sources.append((path, None))
            else:
                sources.extend((child, None) for child in sorted(path.iterdir())
                               if child.is_dir() and (child / 'reports').is_dir())
    if not sources:
        parser.error('No seed runs found')
    try:
        runs = [collect(root, logfile) for root, logfile in sources]
    except (OSError, ValueError) as error:
        parser.error(str(error))
    seeds = [run['seed'] for run in runs]
    if len(seeds) != len(set(seeds)):
        parser.error('Duplicate seed runs selected; supply one retained run per seed')
    runs.sort(key=lambda run: run['seed'])
    warnings = []
    checks = {}
    for label, key in [('identical_split_fingerprints', 'split_sha256'),
                       ('identical_prior', 'prior_sha256'), ('identical_encoding', 'encoding_sha256')]:
        values = [run['preprocessing'].get(key) for run in runs]
        checks[label] = all(v is not None for v in values) and all(v == values[0] for v in values)
        if not checks[label]:
            warnings.append(label + ': unavailable or inconsistent across runs')
    configurations = [run['table_IV'] for run in runs]
    checks['identical_training_configuration'] = all(configurations) and all(v == configurations[0] for v in configurations)
    if not checks['identical_training_configuration']:
        warnings.append('Training configuration unavailable or inconsistent across runs')
    aggregate = {}
    comparable = all(checks.values()) and all(run['full_metrics'].get('N') == 30 for run in runs)
    # Seed spread is descriptive, never a confidence interval or a best-seed selection.
    for metric in METRICS[1:]:
        values = [run['full_metrics'].get(metric) for run in runs]
        if all(finite(v) for v in values):
            aggregate[metric] = {'n_seeds': len(values), 'mean': statistics.mean(values),
                'sample_sd': statistics.stdev(values) if len(values) > 1 else None,
                'comparable_population_and_protocol_verified': comparable}
    report = {'runs': runs, 'across_seed_checks': checks, 'seed_summary': aggregate,
        'warnings': warnings, 'interpretation': 'Mean +/- sample SD describes initialization spread, not a 95% CI. Paired intervals are within-checkpoint row bootstraps. No row-level records, dataframes, checkpoint weights or local paths are exported.'}
    prefix = Path(args.out)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix('.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    columns = ['seed', 'precision', 'selected_epoch', 'selected_VAL_ln_MAE', 'stopping_epoch'] + METRICS
    with prefix.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for run in runs:
            writer.writerow({**{k: run.get(k) for k in columns[:5]}, **{k: run['full_metrics'].get(k) for k in METRICS}})
    lines = ['TIER-3 PAPER UPDATE SUMMARY', 'Seed runs: ' + ', '.join(map(str, sorted(seeds))), '']
    for run in runs:
        lines += [f"[SEED {run['seed']}]", json.dumps({k: run[k] for k in ['precision',
            'selected_epoch', 'selected_VAL_ln_MAE', 'stopping_epoch', 'split_counts', 'full_metrics',
            'comparator_metrics', 'paired_ln_MAE', 'checks', 'runtime', 'warnings']}, indent=2),
            'TABLE IV CONFIGURATION: ' + json.dumps(run['table_IV'], sort_keys=True),
            'TABLE V/VI PREPROCESSING: ' + json.dumps(run['preprocessing'], sort_keys=True), '']
    lines += ['ACROSS-SEED CHECKS: ' + json.dumps(checks),
              'SEED MEAN AND SAMPLE SD: ' + json.dumps(aggregate, indent=2),
              'WARNINGS: ' + json.dumps(warnings), report['interpretation']]
    output = '\n'.join(lines)
    prefix.with_suffix('.txt').write_text(output + '\n')
    print(output)
    print(f'\nWritten: {prefix}.txt, {prefix}.csv, {prefix}.json')


if __name__ == '__main__':
    main()
