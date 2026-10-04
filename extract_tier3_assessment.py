#!/usr/bin/env python3
"""Print compact assessment fields from paper_update_short.txt (stdlib only).
Automatically reads corrected_diagnostics.json beside the TXT if available.
"""
import argparse
import json
import math
import re
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('summary', type=Path, help='Path to paper_update_short.txt')
    args = ap.parse_args()
    source = args.summary.read_text(encoding='utf-8-sig')
    lines = source.splitlines()
    warnings, missing = [], []
    blocks, metrics, paired = {}, {}, {}
    run = {}
    for line in lines:
        line = line.strip()
        match = re.fullmatch(r'N\s*=\s*(\d+)\s*;\s*seed\s*=\s*(\d+)', line)
        if match:
            run.update(N=int(match[1]), seed=int(match[2]))
            continue
        match = re.match(r'^(\w+):\s*(.*)$', line)
        if match and 'ln_MAE=' in match[2]:
            try:
                metrics[match[1]] = {k.strip(): float(v.strip()) for k, v in
                                     (item.split('=', 1) for item in match[2].split(';'))}
            except ValueError:
                warnings.append('Malformed metric line: ' + match[1])
            continue
        if ' = ' not in line:
            continue
        name, value = line.split(' = ', 1)
        if name in ('prior_reconciliation', 'condition_groups', 'prior_checkpoint_sha256'):
            run[name] = value
        elif name.startswith('ln ') or name in ('scalar_controls', 'alignment', 'bound', 'input_output_mapping', 'permutation'):
            try:
                parsed = json.loads(value)
                if not isinstance(parsed, dict):
                    raise ValueError('Expected object')
                target = paired if name.startswith('ln ') else blocks
                if name in target:
                    warnings.append('Duplicate field: ' + name)
                target[name] = parsed
            except ValueError:
                warnings.append('Malformed JSON field: ' + name)

    # TXT lacks k_RMSE and detailed reference checks: obtain them if available.
    sibling = args.summary.with_name('corrected_diagnostics.json')
    extra = {}
    if sibling.exists():
        try:
            extra = json.loads(sibling.read_text(encoding='utf-8-sig'))
            for name, m in extra.get('metrics', {}).items():
                for key, existing in metrics.get(name, {}).items():
                    if key in m and abs(float(existing)-float(m[key])) > 2e-8:
                        raise ValueError('TXT/JSON metrics mismatch')
            for name, m in extra.get('metrics', {}).items():
                for key, value in m.items():
                    metrics.setdefault(name, {}).setdefault(key, value)
            for name in ('input_output_mapping',):
                if name not in blocks and name in extra:
                    blocks[name] = extra[name]
            group_warning = extra.get('condition_groups', {}).get('warning')
            if group_warning:
                warnings.append(group_warning)
        except (ValueError, AttributeError):
            warnings.append('Optional corrected_diagnostics.json could not be read')
            extra = {}

    def val(mapping, key):
        if key not in mapping:
            missing.append(key)
            return 'NOT_REPORTED'
        value = mapping[key]
        if value is None:
            return 'null'
        if isinstance(value, float):
            if not math.isfinite(value):
                warnings.append('Nonfinite value: ' + key)
            return f'{value:.9f}'
        return str(value)

    def fields(mapping, names):
        return '; '.join(key + '=' + val(mapping, key) for key in names)

    print('TIER3 ASSESSMENT — COPY THIS OUTPUT')
    print('RUN: ' + fields(run, ['N', 'seed', 'condition_groups']))
    print('prior_reconciliation = ' + val(run, 'prior_reconciliation'))
    print('prior_checkpoint_sha256 = ' + val(run, 'prior_checkpoint_sha256'))
    for name in ('prior', 'process_off', 'full', 'mean_process_offset', 'TEST_optimal_ln_MAE_scalar'):
        keys = ['ln_MAE', 'ln_RMSE']
        if name == 'full':
            keys += ['k_MAE', 'k_RMSE']
        print(name + ': ' + fields(metrics.get(name, {}), keys))
    requests = {
        'scalar_controls': ['mean_p', 'TEST_optimal_ln_MAE_scalar'],
        'alignment': ['Pearson', 'Spearman'],
        'bound': ['mean_occupancy', 'max_occupancy', 'p_min', 'p_max', 'p_span', 'p_sample_sd'],
        'input_output_mapping': ['unique_process_conditions', 'unique_process_corrections_float32', 'max_within_condition_p_range'],
        'permutation': ['resamples', 'observed_ln_MAE', 'null_mean_ln_MAE', 'null_sample_sd', 'fraction_null_lower_or_equal', 'one_sided_empirical_p'],
    }
    for name, keys in requests.items():
        print(name + ': ' + fields(blocks.get(name, {}), keys))
    for unit in ('row', 'condition_group'):
        for comparator in ('prior', 'process_off', 'mean_process_offset'):
            name = f'ln {comparator}_to_full {unit}'
            item = paired.get(name, {})
            ci = item.get('CI95')
            if isinstance(ci, list) and len(ci) == 2:
                ci_text = '[' + ', '.join(f'{v:.9f}' for v in ci) + ']'
            else:
                ci_text = 'NOT_REPORTED'; missing.append(name + ' CI95')
            print(name + ': delta_MAE=' + val(item, 'delta_MAE') + '; CI95=' + ci_text)
            a = metrics.get(comparator, {}).get('ln_MAE')
            b = metrics.get('full', {}).get('ln_MAE')
            delta = item.get('delta_MAE')
            if all(isinstance(v, (int, float)) for v in [a, b, delta]) and abs(a-b-delta) > 2e-8:
                warnings.append('MAE/delta inconsistency: ' + name)
    reference = extra.get('reference_prior_check', {})
    if 'reference_ln_MAE' in reference:
        print('reference_prior_check: ' + fields(reference, ['status', 'reference_ln_MAE', 'corrected_ln_MAE', 'same_targets', 'max_target_change', 'max_prior_prediction_change']))
    else:
        print('reference_prior_check_details = NOT_AVAILABLE')
    warnings += ['Source reports error/warning: ' + line.strip() for line in lines
                 if re.search(r'\b(traceback|error|warning)\b', line, re.I)]
    print('missing_fields = ' + (', '.join(sorted(set(missing))) or 'NONE'))
    print('extraction_warnings = ' + (' | '.join(warnings) or 'NONE'))
    print('Note: extraction checks do not certify training logs or resolve cross-version prior identity.')


if __name__ == '__main__':
    main()
