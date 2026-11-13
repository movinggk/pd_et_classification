"""Recompute metrics from saved predictions and author a standalone LaTeX report.

No training, network request, model construction, or model forward pass occurs.
The saved tensors are read only to check finiteness and training-only buffers.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile
from zoneinfo import ZoneInfo

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, auc, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, matthews_corrcoef, precision_score, precision_recall_curve,
    roc_auc_score, roc_curve)

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
RUN_ROOT = PROJECT / 'artifacts/pads_mantis_v2_finetuned'
DEFAULT_RUN = RUN_ROOT / '20261109T041659480074Z'
HEAD_ONLY_RUN = RUN_ROOT / '20261109T032315873662Z'
PREVIOUS_RUN = RUN_ROOT / '20261108T145903609065Z'
BASELINES = {'frozen': PROJECT / 'artifacts/pads_mantis_v2',
             'mr_hydra': PROJECT / 'artifacts/pads_mr_hydra'}
LABELS = {'finetuned': 'Last-block', 'head_only': 'Head-only', 'frozen': 'Frozen MantisV2',
          'mr_hydra': 'MR--HYDRA', 'always_pd': 'Always PD'}
METHODS = ['head_only', 'finetuned', 'frozen', 'mr_hydra']
METRICS = ['balanced_accuracy', 'et_recall', 'pd_recall', 'roc_auc',
           'pr_auc_trapezoidal_et', 'average_precision_et', 'g_mean', 'matthews_correlation', 'macro_f1',
           'et_precision', 'f1_et', 'accuracy']


def read_json(path):
    return json.loads(path.read_text())


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b, label):
    if not np.isclose(a, b, rtol=0, atol=1e-12):
        raise ValueError(f'{label}: {a} != {b}')


def require(test, message):
    if not test:
        raise ValueError(message)


def calculate(rows):
    y = np.array([int(r['target_et']) for r in rows])
    s = np.array([float(r['decision_score_et']) for r in rows])
    p = np.array([int(r['predicted_et']) for r in rows])
    require(np.isfinite(s).all() and np.array_equal(p, s > 0), 'Finite logits and zero threshold')
    tn, fp, fn, tp = map(int, confusion_matrix(y, p, labels=[0, 1]).ravel())
    sensitivity, specificity = tp / (tp + fn), tn / (tn + fp)
    precision, recall, _ = precision_recall_curve(y, s)
    close(auc(recall, precision), -np.sum(np.diff(recall) * (precision[:-1] + precision[1:]) / 2),
          'PR-AUC direct trapezoidal sum')
    close(average_precision_score(y, s), -np.sum(np.diff(recall) * precision[:-1]),
          'AP direct stepwise sum')
    return {'balanced_accuracy': float(balanced_accuracy_score(y, p)),
        'et_recall': sensitivity, 'pd_recall': specificity,
        'roc_auc': float(roc_auc_score(y, s)),
        'average_precision_et': float(average_precision_score(y, s)),
        'pr_auc_trapezoidal_et': float(auc(recall, precision)),
        'g_mean': float(np.sqrt(sensitivity * specificity)),
        'matthews_correlation': float(matthews_corrcoef(y, p)),
        'macro_f1': float(f1_score(y, p, average='macro', zero_division=0)),
        'et_precision': float(precision_score(y, p, zero_division=0)),
        'f1_et': float(f1_score(y, p, zero_division=0)),
        'accuracy': float(accuracy_score(y, p)), 'n_pd': tn + fp, 'n_et': fn + tp,
        'pd_correct': tn, 'pd_called_et': fp, 'et_called_pd': fn, 'et_correct': tp,
        'confusion_matrix_true_rows_predicted_columns_pd_et': [[tn, fp], [fn, tp]]}


def compare_saved(metrics, saved, label):
    for k, value in metrics.items():
        if k in ('g_mean', 'pr_auc_trapezoidal_et') or k.startswith(('mean_fold_', 'sd_fold_')):
            # Derived report metrics are not claimed to be original run outputs.
            continue
        if isinstance(value, list):
            require(value == saved[k], f'{label} confusion matrix')
        else:
            close(value, saved[k], f'{label}/{k}')


def paired_summary(predictions):
    """Resample fixed held-out predictions with class sizes and pairing retained."""
    y = np.array([int(r['target_et']) for r in predictions['finetuned']])
    correct = {name: np.array([int(r['predicted_et']) == t for r, t in zip(rows, y)], dtype=float)
               for name, rows in predictions.items()}
    comparisons, paired = {}, {}
    for name in [key for key in ['head_only', 'frozen', 'mr_hydra'] if key in predictions]:
        delta = correct['finetuned'] - correct[name]
        rng = np.random.default_rng(20261109)
        ba_draws, recalls = np.zeros(20000), {}
        paired[name] = {}
        for cls, clabel in [(0, 'PD'), (1, 'ET')]:
            idx = np.flatnonzero(y == cls)
            draws = delta[rng.choice(idx, size=(20000, len(idx)), replace=True)].mean(axis=1)
            ba_draws += draws / 2
            recalls[clabel] = np.quantile(draws, [0.025, 0.975]).tolist()
            a, b = correct[name][idx].astype(bool), correct['finetuned'][idx].astype(bool)
            paired[name][clabel] = {'both_correct': int((a & b).sum()),
                'only_finetuned_correct': int((~a & b).sum()),
                'only_baseline_correct': int((a & ~b).sum()), 'both_wrong': int((~a & ~b).sum())}
        comparisons[name] = {'delta_ba': float(delta[y == 0].mean() / 2 + delta[y == 1].mean() / 2),
            'delta_ba_percentile_95': np.quantile(ba_draws, [0.025, 0.975]).tolist(),
            'delta_recall_percentile_95': recalls}
    return {'samples': 20000, 'seed': 20261109, 'class_sizes': {'PD': 276, 'ET': 28},
            'comparisons': comparisons,
            'limitation': 'Conditional on saved fits; no training-set, model-selection or external-cohort uncertainty.'}, paired


def audit(run):
    manifest, summary, plan = [read_json(run / f) for f in ['manifest.json', 'summary.json', 'splits.json']]
    require(manifest['status'] == 'complete', 'Run must be complete')
    cfg = manifest['config']
    require(cfg['mode'] in ['head-only', 'last-block'], 'Supported report modes')
    encoder_lrs = [0.0] if cfg['mode'] == 'head-only' else [1e-5, 3e-5]
    expected = dict(input_set='all', outer_folds=5, inner_folds=3,
        repeats=1, seed=20261017, epochs=[3, 5, 10], encoder_lrs=encoder_lrs,
        weight_decays=[0.01], head_lr=1e-4, head_dropout=0.1, head_warmup_epochs=1,
        batch_size=2, accumulation_steps=8, channel_batch_size=16,
        activation_checkpointing=True, save_outer_models=True, fit_final=True)
    require(all(cfg[k] == v for k, v in expected.items()), 'Template protocol must match completed run')
    require(manifest['parameter_counts'] == {'stored_total': 2281729,
        'encoder_trainable': 0 if cfg['mode'] == 'head-only' else 658176,
        'head_trainable': 67585}, 'Mode-specific parameter counts')
    cohort = read_csv(run / 'cohort.csv')
    ids = [r['participant_id'] for r in cohort]
    y = np.array([int(r['target_et']) for r in cohort])
    target = dict(zip(ids, y.tolist()))
    require(len(ids) == len(set(ids)) == 304 and int(y.sum()) == 28, '304 participants, 28 ET')
    require(all(r['condition'] == ("Parkinson's" if target[r['participant_id']] == 0
                                  else 'Essential Tremor') for r in cohort), 'Exact diagnosis mapping')
    require(sum(r['pd_with_et_comment'] == 'True' for r in cohort) == 8, 'Eight retained PD comment cases')
    require(len(plan) == 5, 'Five outer folds')
    outer_seen, split_counts, inner_counts = [], [], []
    for fold, s in enumerate(plan, 1):
        require(s['repeat'] == 1 and s['fold'] == fold, 'Fold order')
        tr, te = set(s['train_ids']), set(s['test_ids'])
        require(not tr & te and tr | te == set(ids), 'Outer partition')
        require([ids[i] for i in s['train_indices']] == s['train_ids'], 'Training index-ID match')
        require([ids[i] for i in s['test_indices']] == s['test_ids'], 'Test index-ID match')
        require(len(s['inner_splits']) == 3, 'Three inner folds')
        valid_seen = []
        for z in s['inner_splits']:
            a, b = set(z['train_ids']), set(z['validation_ids'])
            require(not a & b and a | b == tr, 'Inner partition excludes outer test')
            require([s['train_ids'][i] for i in z['train_positions']] == z['train_ids'], 'Inner train indices')
            require([s['train_ids'][i] for i in z['validation_positions']] == z['validation_ids'], 'Inner valid indices')
            valid_seen.extend(z['validation_ids'])
            inner_counts.append({'training_et': sum(target[i] for i in a),
                                 'validation_et': sum(target[i] for i in b)})
        require(sorted(valid_seen) == sorted(tr), 'Exactly one inner validation per training participant')
        outer_seen.extend(s['test_ids'])
        split_counts.append({'fold': fold, 'train_pd': sum(target[i] == 0 for i in tr),
            'train_et': sum(target[i] for i in tr), 'test_pd': sum(target[i] == 0 for i in te),
            'test_et': sum(target[i] for i in te), 'model_seed': s['model_seed']})
    require(sorted(outer_seen) == sorted(ids), 'Exactly one outer test per participant')

    roots = {'finetuned': run, **BASELINES}
    results, folds, predictions = {}, {}, {}
    for name, root in roots.items():
        m = read_json(root / 'manifest.json')
        require(m['status'] == 'complete', f'{name} complete')
        require(read_csv(root / 'cohort.csv') == cohort, f'{name} identical ordered cohort')
        require(read_json(root / 'splits.json') == plan, f'{name} identical nested splits')
        for key in ['verified_source_sha256', 'checksum_list_sha256', 'movement_shape', 'channel_names', 'label_mapping']:
            require(m['dataset'][key] == manifest['dataset'][key], f'{name} data provenance {key}')
        rows = read_csv(root / 'oof_predictions.csv')
        require(len(rows) == 304 and len({r['participant_id'] for r in rows}) == 304, f'{name} coverage')
        byid = {r['participant_id']: r for r in rows}
        for r in rows:
            require(r['participant_id'] in target and int(r['target_et']) == target[r['participant_id']], f'{name} labels')
            require(int(r['repeat']) == 1 and r['participant_id'] in plan[int(r['fold']) - 1]['test_ids'], f'{name} held-out identity')
        predictions[name] = [byid[i] for i in ids]
        results[name] = calculate(predictions[name])
        saved = read_json(root / 'summary.json')
        compare_saved(results[name], saved['repeat_metrics'][0], name)
        folds[name] = []
        table = read_csv(root / 'fold_metrics.csv')
        for k in range(1, 6):
            subset = [r for r in rows if int(r['fold']) == k]
            val = {'fold': k, **calculate(subset)}
            compare_saved(val, saved['outer_fold_metrics'][k - 1], f'{name}/fold{k}')
            for key in METRICS:
                if key not in ('g_mean', 'pr_auc_trapezoidal_et'):
                    close(val[key], float(table[k - 1][key]), 'Fold CSV')
            folds[name].append(val)
        results[name]['mean_fold_roc_auc'] = float(np.mean([r['roc_auc'] for r in folds[name]]))
        results[name]['mean_fold_ap'] = float(np.mean([r['average_precision_et'] for r in folds[name]]))
        results[name]['mean_fold_pr_auc_trapezoidal_et'] = float(np.mean([r['pr_auc_trapezoidal_et'] for r in folds[name]]))
        results[name]['mean_fold_ba'] = float(np.mean([r['balanced_accuracy'] for r in folds[name]]))
        results[name]['sd_fold_ba_descriptive'] = float(np.std([r['balanced_accuracy'] for r in folds[name]], ddof=1))
    for key in ['roc_auc', 'average_precision_et']:
        close(np.mean([r[key] for r in folds['finetuned']]), summary['mean_outer_fold_' + key], 'Fine-tuned fold mean')
    always = [dict(r, decision_score_et='-1', predicted_et='0') for r in predictions['finetuned']]
    results['always_pd'] = calculate(always)
    compare_saved(results['always_pd'], summary['always_pd_baseline'], 'Always PD')

    # Verify saved initial representations and model normalization without any forward pass.
    provenance = read_json(run / 'initial_encoder_provenance.json')
    frozen_prov = read_json(BASELINES['frozen'] / 'encoder_provenance.json')
    for key in ['encoder', 'input_shape', 'feature_shape', 'parameters_trainable', 'representation']:
        require(provenance[key] == frozen_prov[key], f'Frozen representation identity {key}')
    for fname, expected_hash in provenance['checkpoint']['sha256'].items():
        require(sha(run / 'initial_checkpoint' / fname) == expected_hash, f'Initial checkpoint {fname}')
        require(expected_hash == frozen_prov['checkpoint']['sha256'][fname], 'Same pretrained weights')
    Z0 = np.load(run / 'initial_frozen_features.npy', allow_pickle=False)
    require(Z0.shape == (304, 67584) and Z0.dtype == np.float32 and np.isfinite(Z0).all(), 'Initial feature dimensions')
    require(sha(run / 'initial_frozen_features.npy') == provenance['feature_sha256'], 'Initial features checksum')
    from safetensors.torch import load_file
    initial_state = load_file(str(run / 'initial_checkpoint/model.safetensors'))
    choices, searches, histories, model_checks = [], [], [], []

    def audit_fit(root, train_ids, inner, prefix='', fold=None):
        selected = read_json(root / (prefix + 'selected.json'))
        search = read_csv(root / (prefix + 'inner_search.csv'))
        require(len(search) == len(encoder_lrs) * 3, 'Candidate count matches mode')
        typed = [{k: int(v) if k == 'epochs' else float(v) for k, v in row.items()} for row in search]
        require({(r['encoder_lr'], r['weight_decay'], r['epochs']) for r in typed} ==
            {(lr, 0.01, e) for lr in encoder_lrs for e in [3, 5, 10]}, 'Candidate grid')
        best = max(r['inner_mean_balanced_accuracy'] for r in typed)
        chosen = min([r for r in typed if abs(r['inner_mean_balanced_accuracy'] - best) <= 1e-12],
                     key=lambda r: (-r['weight_decay'], r['epochs'], r['encoder_lr']))
        for k, v in selected.items(): close(v, chosen[k], 'Selected candidate')
        inner_rows = read_csv(root / (prefix + 'inner_oof_predictions.csv'))
        require([r['participant_id'] for r in inner_rows] == train_ids, 'Selected inner coverage/order')
        require(all(int(r['target_et']) == target[r['participant_id']] for r in inner_rows), 'Inner labels')
        inner_metrics = calculate(inner_rows)
        close(inner_metrics['balanced_accuracy'], selected['inner_pooled_balanced_accuracy'], 'Selected inner pooled BA')
        close(inner_metrics['roc_auc'], selected['inner_pooled_roc_auc'], 'Selected inner pooled AUC')
        byid = {r['participant_id']: r for r in inner_rows}
        bas = [calculate([byid[i] for i in z['validation_ids']])['balanced_accuracy'] for z in inner]
        close(np.mean(bas), selected['inner_mean_balanced_accuracy'], 'Selected mean inner BA')
        close(np.std(bas, ddof=1), selected['inner_std_balanced_accuracy_descriptive'], 'Selected inner SD')
        history = read_csv(root / (prefix + 'training_history.csv'))
        require([int(r['epoch']) for r in history] == list(range(1, selected['epochs'] + 1)), 'Refit epoch count')
        require(all(r['head_warmup'] == str(cfg['mode'] != 'head-only' and int(r['epoch']) == 1)
                    and np.isfinite(float(r['training_weighted_bce'])) for r in history), 'Mode-specific warmup and loss')
        model_path = root / ('final_model.pt' if prefix else 'model.pt')
        payload = torch.load(model_path, map_location='cpu', weights_only=True)
        require(payload['model_type'] == 'mantis_v2_finetuned' and payload['format_version'] == 1,
                'Bundle format')
        require(payload['metadata']['training_ids'] == train_ids, 'Saved model training IDs')
        require(payload['metadata']['selected'] == selected, 'Saved model selected settings')
        require(payload['metadata']['source_code_sha256'] == manifest['source_code_sha256'], 'Saved model source identity')
        require(payload['model_config'] == manifest['model'] and payload['input_set'] == 'all'
                and payload['decision_threshold'] == 0, 'Model input protocol')
        require(payload['initial_checkpoint']['sha256'] == provenance['checkpoint']['sha256'], 'Bundle initial weights identity')
        state = payload['state_dict']
        require(all(torch.isfinite(v).all().item() for v in state.values()), 'Finite saved tensors')
        frozen_tensors, adapted_tensors = 0, 0
        for key, tensor in state.items():
            if not key.startswith('encoder.'):
                continue
            original = initial_state[key.removeprefix('encoder.')]
            adapted = cfg['mode'] == 'last-block' and key.startswith('encoder.transf_unit.transformer.layers.2.')
            if adapted:
                adapted_tensors += int(not torch.equal(tensor, original))
            else:
                require(torch.equal(tensor, original), f'Frozen encoder tensor unchanged: {key}')
                frozen_tensors += 1
        if cfg['mode'] == 'last-block':
            require(adapted_tensors > 0, 'Last active block actually adapted')
        index = np.array([ids.index(i) for i in train_ids])
        mean = np.mean(Z0[index], axis=0, dtype=np.float64).astype(np.float32)
        scale = np.std(Z0[index], axis=0, dtype=np.float64)
        scale[scale < 1e-6] = 1.0
        require(np.array_equal(state['feature_mean'].numpy(), mean), 'Training-only feature mean')
        require(np.array_equal(state['feature_scale'].numpy(), scale.astype(np.float32)), 'Training-only feature scale')
        train_index = [i for i in state if i.startswith('encoder.transf_unit.transformer.layers.')]
        require({int(k.split('.')[4]) for k in train_index} == {0, 1, 2}, 'Exactly three stored active blocks')
        check = {'fold': fold, 'path': str(model_path.relative_to(PROJECT)), 'sha256': sha(model_path),
            'bytes': model_path.stat().st_size, 'state_tensor_count': len(state),
            'train_only_statistics_exact': True, 'all_tensors_finite': True,
            'frozen_encoder_tensors_verified': frozen_tensors,
            'adapted_third_block_tensors_changed': adapted_tensors,
            'optimizer_resume_state_present': 'optimizer' in payload}
        model_checks.append(check)
        choices.append({'mode': cfg['mode'], 'fold': fold, **selected})
        searches.extend({'mode': cfg['mode'], 'fold': fold, **r} for r in typed)
        histories.append({'mode': cfg['mode'], 'fold': fold, 'epochs': history})
        return payload

    for s in plan:
        root = run / f"repeat_01_fold_{s['fold']:02d}"
        audit_fit(root, s['train_ids'], s['inner_splits'], fold=s['fold'])
        test_rows = read_csv(root / 'test_predictions.csv')
        require({r['participant_id']: r for r in test_rows} ==
                {r['participant_id']: r for r in predictions['finetuned'] if int(r['fold']) == s['fold']}, 'Fold predictions equal aggregate')
        compare_saved(folds['finetuned'][s['fold'] - 1], read_json(root / 'metrics.json'), 'Saved fold metrics')
    final_inner = read_json(run / 'final_inner_splits.json')
    final_seen = []
    require(len(final_inner) == 3, 'Three final selection folds')
    for s in final_inner:
        a, b = set(s['train_ids']), set(s['validation_ids'])
        require(not a & b and a | b == set(ids), 'Final selection partition')
        final_seen.extend(s['validation_ids'])
    require(sorted(final_seen) == sorted(ids), 'Final selection coverage')
    audit_fit(run, ids, final_inner, prefix='final_')

    # Check every selected source data hash, and the archived code actually evaluated.
    data_root = Path(cfg['data_root'])
    source_hashes = manifest['dataset']['verified_source_sha256']
    for path, expected_hash in source_hashes.items():
        require(sha(data_root / path) == expected_hash, f'Dataset hash {path}')
    with zipfile.ZipFile(run / 'evaluated_sources.zip') as archive:
        require(archive.testzip() is None, 'Evaluated code archive CRC')
        for path, expected_hash in manifest['source_code_sha256'].items():
            require(hashlib.sha256(archive.read(path)).hexdigest() == expected_hash, f'Evaluated source {path}')
    current_match = {p: sha(PROJECT / p) == h for p, h in manifest['source_code_sha256'].items()}
    comp = read_json(run / 'frozen_comparison.json')['repeat_comparisons'][0]
    for name in ['frozen', 'finetuned']: compare_saved(results[name], comp[name], 'Comparison metrics')
    for k, v in comp['finetuned_minus_frozen'].items():
        close(results['finetuned'][k] - results['frozen'][k], v, 'Comparison delta')

    bootstrap, paired = paired_summary(predictions)

    artifact_names = ['manifest.json', 'summary.json', 'splits.json', 'cohort.csv', 'oof_predictions.csv',
        'fold_metrics.csv', 'frozen_comparison.json', 'initial_encoder_provenance.json',
        'initial_frozen_features.npy', 'evaluated_sources.zip', 'final_selected.json',
        'final_inner_splits.json', 'final_inner_oof_predictions.csv', 'final_inner_search.csv',
        'final_training_history.csv']
    for s in plan:
        for n in ['selected.json', 'inner_search.csv', 'inner_oof_predictions.csv',
                  'training_history.csv', 'test_predictions.csv', 'metrics.json']:
            artifact_names.append(f"repeat_01_fold_{s['fold']:02d}/{n}")
    return {'run_id': run.name, 'run_relative': str(run.relative_to(PROJECT)),
        'audited_at_utc': datetime.now(timezone.utc).isoformat(),
        'no_training_or_model_forward_pass': True, 'manifest': manifest,
        'results': results, 'fold_metrics': folds, 'selected_settings': choices,
        'search_results': searches, 'training_history': histories,
        'split_counts': split_counts, 'inner_counts': inner_counts,
        'model_checks': model_checks, 'paired_outcomes': paired,
        'conditional_paired_bootstrap': bootstrap,
        'provenance_checks': {'verified_dataset_files': len(source_hashes),
            'verified_archived_code_files': len(manifest['source_code_sha256']),
            'current_code_matches_evaluated': current_match,
            'identical_three_method_nested_splits': True, 'initial_checkpoint_verified': True},
        'baseline_artifact_sha256': {name: {fname: sha(root / fname) for fname in
            ['manifest.json', 'summary.json', 'cohort.csv', 'splits.json', 'oof_predictions.csv', 'fold_metrics.csv']}
            for name, root in BASELINES.items()},
        'artifact_sha256': {p: sha(run / p) for p in artifact_names}}, predictions


def audit_all(last_run, head_run, previous_run):
    last, predictions = audit(last_run)
    head, hp = audit(head_run)
    previous, pp = audit(previous_run)
    require(last['manifest']['config']['mode'] == previous['manifest']['config']['mode'] == 'last-block',
            'Primary and previous runs must be last-block')
    require(head['manifest']['config']['mode'] == 'head-only', 'Head-only control must freeze encoder')
    shared = lambda cfg: {k: v for k, v in cfg.items() if k not in ['mode', 'encoder_lrs']}
    require(shared(last['manifest']['config']) == shared(head['manifest']['config']) ==
            shared(previous['manifest']['config']), 'All other execution settings identical')
    for candidate in [head_run, previous_run]:
        for fname in ['cohort.csv', 'splits.json', 'final_inner_splits.json', 'initial_frozen_features.npy']:
            require(sha(last_run / fname) == sha(candidate / fname), f'Cross-run identity: {fname}')
    require(last['manifest']['source_code_sha256'] == head['manifest']['source_code_sha256'] ==
            previous['manifest']['source_code_sha256'], 'All three evaluated source snapshots identical')
    require(predictions['finetuned'] == pp['finetuned'], 'Repeated last-block predictions exactly identical')
    require(last['selected_settings'] == previous['selected_settings'] and
            last['search_results'] == previous['search_results'] and
            last['training_history'] == previous['training_history'], 'Repeated selection and training histories identical')
    repeated_models = []
    for old_check, new_check in zip(previous['model_checks'], last['model_checks']):
        old_path, new_path = PROJECT / old_check['path'], PROJECT / new_check['path']
        old_state = torch.load(old_path, map_location='cpu', weights_only=True)['state_dict']
        new_state = torch.load(new_path, map_location='cpu', weights_only=True)['state_dict']
        require(old_state.keys() == new_state.keys() and all(torch.equal(old_state[k], new_state[k]) for k in old_state),
                'Repeated last-block saved tensors exactly identical')
        repeated_models.append({'fold': new_check['fold'], 'all_state_tensors_identical': True,
                                'serialized_file_hash_identical': old_check['sha256'] == new_check['sha256']})
    last['additional_runs'] = {'head_only': head, 'previous_last_block': previous}
    last['results']['head_only'] = head['results']['finetuned']
    last['fold_metrics']['head_only'] = head['fold_metrics']['finetuned']
    predictions['head_only'] = hp['finetuned']
    last['selected_settings'] = head['selected_settings'] + last['selected_settings']
    last['search_results'] = head['search_results'] + last['search_results']
    last['training_history'] = head['training_history'] + last['training_history']
    last['conditional_paired_bootstrap'], last['paired_outcomes'] = paired_summary(predictions)
    last['cross_run_verification'] = {'matched_inputs_cohort_inner_outer_and_final_splits': True,
        'identical_evaluated_code': True, 'same_seed_last_block_repeat': {
            'previous_run_id': previous_run.name, 'latest_run_id': last_run.name,
            'all_outer_predictions_exactly_identical': True,
            'selection_and_training_history_exactly_identical': True, 'models': repeated_models,
            'independent_additional_test_participants': 0}}
    last['run_registry'] = [{'role': role, 'run_id': obj['run_id'], 'mode': obj['manifest']['config']['mode'],
        'started_at_utc': obj['manifest']['started_at_utc'], 'finished_at_utc': obj['manifest']['finished_at_utc'],
        'elapsed_seconds': obj['manifest']['elapsed_seconds']}
        for role, obj in [('head-only', head), ('last-block', last), ('previous last-block', previous)]]
    return last, predictions


def num(value):
    return f'{value:.4f}'


def pct(value):
    return f'{100 * value:.2f}'


def coords(x, y):
    return ' '.join(f'({a:.7f},{b:.7f})' for a, b in zip(x, y))


def author(verified, predictions):
    r = verified['results']
    replacements = {'RUN_ID': verified['run_id'],
                    'HEAD_ONLY_RUN_ID': verified['additional_runs']['head_only']['run_id']}
    replacements['RUN_TABLE'] = '\n'.join(
        ('Head-only' if z['role'] == 'head-only' else 'Last-block' if z['role'] == 'last-block' else 'Last-block (old)') +
        r' & \code{' + z['run_id'] + '} & ' +
        datetime.fromisoformat(z['started_at_utc']).astimezone(ZoneInfo('Asia/Seoul')).strftime('%m-%d %H:%M:%S') + ' & ' +
        datetime.fromisoformat(z['finished_at_utc']).astimezone(ZoneInfo('Asia/Seoul')).strftime('%m-%d %H:%M:%S') + ' & ' +
        f"{z['elapsed_seconds']/60:.2f}" + r' \\' for z in verified['run_registry'])
    replacements['MAIN_TABLE'] = '\n'.join(
        LABELS[name] + ' & ' + ' & '.join(pct(r[name][k]) for k in ['balanced_accuracy', 'et_recall', 'pd_recall']) +
        ' & ' + num(r[name]['roc_auc']) + ' & ' +
        (num(r[name]['pr_auc_trapezoidal_et']) if name != 'always_pd' else '--') + r' \\'
        for name in METHODS + ['always_pd'])
    replacements['IMBALANCE_TABLE'] = '\n'.join(
        LABELS[name] + ' & ' + ' & '.join(num(r[name][k]) for k in ['g_mean', 'matthews_correlation', 'macro_f1', 'f1_et']) +
        ' & ' + num(r[name]['average_precision_et']) + ' & ' + pct(r[name]['et_precision']) + r' \\'
        for name in METHODS + ['always_pd'])
    replacements['FOLD_MEAN_TABLE'] = '\n'.join(
        LABELS[name] + ' & ' + pct(r[name]['mean_fold_ba']) + ' & ' + pct(r[name]['sd_fold_ba_descriptive']) +
        ' & ' + num(r[name]['mean_fold_roc_auc']) + ' & ' + num(r[name]['mean_fold_pr_auc_trapezoidal_et']) +
        ' & ' + num(r[name]['mean_fold_ap']) + r' \\'
        for name in METHODS)
    replacements['FOLD_TABLE'] = '\n'.join(
        LABELS[name] + f" & {f['fold']} & {f['pd_correct']}/{f['n_pd']} & {f['et_correct']}/{f['n_et']} & " +
        ' & '.join([pct(f['balanced_accuracy']), num(f['roc_auc']), num(f['pr_auc_trapezoidal_et']),
                    num(f['g_mean']), num(f['matthews_correlation'])]) + r' \\'
        for name in ['head_only', 'finetuned'] for f in verified['fold_metrics'][name])
    replacements['SELECTION_TABLE'] = '\n'.join(
        ('Head-only' if s['mode'] == 'head-only' else 'Last-block') + ' & ' +
        (str(s['fold']) if s['fold'] else 'Final') + ' & ' + f"{s['encoder_lr']:.0e}" + ' & ' +
        str(s['epochs']) + ' & ' + pct(s['inner_mean_balanced_accuracy']) + ' & ' +
        pct(s['inner_std_balanced_accuracy_descriptive']) + r' \\'
        for s in verified['selected_settings'])
    replacements['SPLIT_TABLE'] = '\n'.join(
        ' & '.join(str(s[k]) for k in ['fold', 'train_pd', 'train_et', 'test_pd', 'test_et', 'model_seed']) + r' \\'
        for s in verified['split_counts'])
    replacements['SEARCH_TABLE'] = '\n'.join(
        ('Head' if s['mode'] == 'head-only' else 'Last') + ' & ' +
        (str(s['fold']) if s['fold'] else 'Final') + ' & ' + f"{s['encoder_lr']:.0e}" + ' & ' +
        str(s['epochs']) + ' & ' + pct(s['inner_mean_balanced_accuracy']) + ' & ' +
        pct(s['inner_std_balanced_accuracy_descriptive']) + ' & ' + num(s['inner_pooled_roc_auc']) + r' \\'
        for s in verified['search_results'])
    replacements['PAIRED_TABLE'] = '\n'.join(
        LABELS[name] + f" / {cls} & " + ' & '.join(str(d[k]) for k in
            ['both_correct', 'only_finetuned_correct', 'only_baseline_correct', 'both_wrong']) + r' \\'
        for name, classes in verified['paired_outcomes'].items() for cls, d in classes.items())
    replacements['BOOTSTRAP_TABLE'] = '\n'.join(
        LABELS[name] + ' & ' + pct(d['delta_ba']) + ' & [' +
        ', '.join(pct(x) for x in d['delta_ba_percentile_95']) + ']' + r' \\'
        for name, d in verified['conditional_paired_bootstrap']['comparisons'].items())
    for name in METHODS:
        rows = predictions[name]
        y, s = [int(x['target_et']) for x in rows], [float(x['decision_score_et']) for x in rows]
        fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False)
        precision, recall, _ = precision_recall_curve(y, s)
        replacements['ROC_' + name.upper()] = coords(fpr, tpr)
        replacements['PR_' + name.upper()] = coords(recall[::-1], precision[::-1])
        replacements['BA_' + name.upper()] = coords(range(1, 6), [100 * z['balanced_accuracy'] for z in verified['fold_metrics'][name]])
        replacements['AUC_' + name.upper()] = coords(range(1, 6), [z['roc_auc'] for z in verified['fold_metrics'][name]])
    replacements['TRAINING_PLOTS'] = '\n'.join(
        r'\addplot+[mark=*,thick] coordinates {' + coords([int(x['epoch']) for x in h['epochs']],
                   [float(x['training_weighted_bce']) for x in h['epochs']]) + r'};\addlegendentry{Fold ' + str(h['fold']) + '}'
        for h in verified['training_history'] if h['fold'] and h['mode'] == 'last-block')
    replacements['HEAD_TRAINING_PLOTS'] = '\n'.join(
        r'\addplot+[mark=*,thick] coordinates {' + coords([int(x['epoch']) for x in h['epochs']],
                   [float(x['training_weighted_bce']) for x in h['epochs']]) + r'};\addlegendentry{Fold ' + str(h['fold']) + '}'
        for h in verified['training_history'] if h['fold'] and h['mode'] == 'head-only')
    template = (HERE / 'report_template.tex').read_text()
    for k, v in replacements.items(): template = template.replace('@@' + k + '@@', v)
    require('@@' not in template, 'No unresolved LaTeX placeholders')
    (HERE / 'report.tex').write_text(template)
    with (HERE / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['method', 'method_label', 'run_id'] + METRICS)
        writer.writeheader()
        for name in METHODS + ['always_pd']:
            metrics = r[name]
            run_id = verified['run_id'] if name == 'finetuned' else replacements['HEAD_ONLY_RUN_ID'] if name == 'head_only' else ''
            writer.writerow({'method': name, 'method_label': LABELS[name], 'run_id': run_id,
                             **{k: metrics[k] for k in METRICS}})
    with (HERE / 'fold_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['method', 'method_label', 'fold'] + METRICS)
        writer.writeheader()
        for name in METHODS:
            for row in verified['fold_metrics'][name]:
                writer.writerow({'method': name, 'method_label': LABELS[name], 'fold': row['fold'],
                                 **{k: row[k] for k in METRICS}})
    (HERE / 'verification.json').write_text(json.dumps(verified, indent=2, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--head-only-run', type=Path, default=HEAD_ONLY_RUN)
    parser.add_argument('--previous-run', type=Path, default=PREVIOUS_RUN)
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    verified, predictions = audit_all(args.run.resolve(), args.head_only_run.resolve(), args.previous_run.resolve())
    if not args.audit_only: author(verified, predictions)
    print(json.dumps({'audit': 'passed', 'model_forward_passes': 0, 'training_runs': 0,
        'results': verified['results'], 'paired': verified['paired_outcomes'],
        'conditional_bootstrap': verified['conditional_paired_bootstrap'],
        'provenance': verified['provenance_checks']}, indent=2))


if __name__ == '__main__':
    main()
