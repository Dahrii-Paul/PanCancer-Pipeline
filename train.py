"""
train.py — unified pipeline for all three classification tasks.

Usage:
    python src/train.py --task multiclass
    python src/train.py --task three_class
    python src/train.py --task binary
    python src/train.py --task all          # runs all three in sequence
"""

import os
import time
import argparse

import numpy as np
import pandas as pd
from pycaret.classification import (
    setup, create_model, tune_model,
    predict_model, pull, add_metric, save_model
)

from config import (
    BASE_OUTPUT_DIR,
    PYCARET_CONFIG,
    MODELS_LIST,
    TASK_CONFIGS
)
from evaluate import (
    multiclass_specificity,
    compute_classwise_metrics,
    save_confusion_matrix
)


# ── Directory helpers ─────────────────────────────────────────────────────────
def make_dirs(task_label):
    """Create output directories for a given task label."""
    dirs = {
        'output':    f"{BASE_OUTPUT_DIR}/{task_label}",
        'models':    f"{BASE_OUTPUT_DIR}/{task_label}/saved_models",
        'cm':        f"{BASE_OUTPUT_DIR}/{task_label}/confusion_matrices",
        'classwise': f"{BASE_OUTPUT_DIR}/{task_label}/classwise_metrics",
        'results':   f"./results/{task_label}"
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data(task_cfg):
    target_col     = task_cfg['target_col']
    target_mapping = task_cfg['target_mapping']

    df_train = pd.read_csv(task_cfg['train_path'])
    df_test  = pd.read_csv(task_cfg['test_path'])

    X_train = df_train.drop(target_col, axis=1)
    y_train = df_train[target_col].map(target_mapping)
    X_test  = df_test.drop(target_col, axis=1)
    y_test  = df_test[target_col].map(target_mapping)

    train_mapped = pd.concat([X_train, y_train.rename(target_col)], axis=1)
    test_mapped  = pd.concat([X_test,  y_test.rename(target_col)],  axis=1)

    print(f"  Train: {df_train.shape} | Test: {df_test.shape}")
    return X_train, y_train, X_test, y_test, train_mapped, test_mapped


# ── PyCaret experiment setup ──────────────────────────────────────────────────
def init_experiment(train_mapped, target_col):
    setup(data=train_mapped, target=target_col, **PYCARET_CONFIG)
    add_metric(
        id='spec',
        name='Specificity',
        score_func=multiclass_specificity,
        target='pred',
        greater_is_better=True
    )


# ── Single model training (original + tuned) ──────────────────────────────────
def run_model(model_id, task_label, class_names,
              X_test, y_test, test_mapped, dirs,
              all_classwise, mean_results, mean_results_tuned, all_results,
              n_trials=10):

    n_classes = len(class_names)
    print(f"\n  ----- {model_id} -----")

    # ── Original ──────────────────────────────────────────────────────────────
    start = time.time()
    kwargs = {'class_weight': 'balanced'} if model_id in ('rf', 'et') else {}
    created_model = create_model(model_id, **kwargs)
    training_time = time.time() - start

    model_name = f"{model_id}_{task_label}_original"
    save_model(created_model, f"{dirs['models']}/{model_name}")
    print(f"    Saved: {model_name}  ({training_time:.2f}s)")

    result = pull()
    result.loc['Mean', 'Model'] = model_id
    mean_result = result.loc['Mean'].copy()
    mean_result['Training_Time'] = training_time
    mean_results.append(mean_result)

    test_pred_df = predict_model(created_model, data=test_mapped, raw_score=True)
    model_test   = pull()
    model_test['Type']          = 'test'
    model_test['Model']         = model_name
    model_test['Training_Time'] = training_time

    y_pred_orig = test_pred_df['prediction_label'].values
    prob_cols   = sorted([c for c in test_pred_df.columns if 'prediction_score_' in c])
    y_prob_orig = (test_pred_df[prob_cols].values
                   if prob_cols else np.zeros((len(y_test), n_classes)))

    save_confusion_matrix(y_test, y_pred_orig, class_names, model_name, dirs['cm'])

    cw_orig = compute_classwise_metrics(y_test, y_pred_orig, y_prob_orig, class_names, model_name)
    cw_orig.to_csv(f"{dirs['classwise']}/{model_name}_classwise.csv", index=False)
    all_classwise = pd.concat([all_classwise, cw_orig], axis=0)

    # ── Tuned ─────────────────────────────────────────────────────────────────
    print(f"    Tuning {model_id}...")
    start = time.time()
    tuned_model = tune_model(created_model, search_library='optuna', n_iter=n_trials)
    tuning_time = time.time() - start

    tuned_name = f"{model_id}_{task_label}_tuned"
    save_model(tuned_model, f"{dirs['models']}/{tuned_name}")
    print(f"    Saved: {tuned_name}  ({tuning_time:.2f}s)")

    result_tuned = pull()
    result_tuned.loc['Mean', 'Model'] = tuned_name
    mean_result_tuned = result_tuned.loc['Mean'].copy()
    mean_result_tuned['Training_Time'] = tuning_time
    mean_results_tuned.append(mean_result_tuned)

    tuned_pred_df = predict_model(tuned_model, data=test_mapped, raw_score=True)
    model_test_tuned   = pull()
    model_test_tuned['Type']          = 'test'
    model_test_tuned['Model']         = tuned_name
    model_test_tuned['Training_Time'] = tuning_time

    y_pred_tuned = tuned_pred_df['prediction_label'].values
    prob_cols_t  = sorted([c for c in tuned_pred_df.columns if 'prediction_score_' in c])
    y_prob_tuned = (tuned_pred_df[prob_cols_t].values
                    if prob_cols_t else np.zeros((len(y_test), n_classes)))

    save_confusion_matrix(y_test, y_pred_tuned, class_names, tuned_name, dirs['cm'])

    cw_tuned = compute_classwise_metrics(y_test, y_pred_tuned, y_prob_tuned, class_names, tuned_name)
    cw_tuned.to_csv(f"{dirs['classwise']}/{tuned_name}_classwise.csv", index=False)
    all_classwise = pd.concat([all_classwise, cw_tuned], axis=0)

    # ── Per-model combined results ─────────────────────────────────────────────
    combined_df = pd.concat([
        mean_result.to_frame().T.assign(Type='train', Model=model_id),
        model_test,
        mean_result_tuned.to_frame().T.assign(Type='train', Model=tuned_name),
        model_test_tuned
    ], axis=0)

    combined_df.to_csv(f"{dirs['results']}/{model_id}_results.csv", index=False)
    all_results = pd.concat([all_results, combined_df], axis=0)

    return all_classwise, mean_results, mean_results_tuned, all_results


# ── Full pipeline for one task ────────────────────────────────────────────────
def run_task(task_name, n_trials=10):
    task_cfg   = TASK_CONFIGS[task_name]
    task_label = task_cfg['label']
    class_names = list(task_cfg['target_mapping'].keys())

    print(f"\n{'='*60}")
    print(f"  TASK: {task_name.upper()}  ({len(class_names)} classes)")
    print(f"{'='*60}")

    dirs = make_dirs(task_label)

    X_train, y_train, X_test, y_test, train_mapped, test_mapped = load_data(task_cfg)
    init_experiment(train_mapped, task_cfg['target_col'])

    all_classwise      = pd.DataFrame()
    all_results        = pd.DataFrame()
    mean_results       = []
    mean_results_tuned = []

    for model_id in MODELS_LIST:
        try:
            all_classwise, mean_results, mean_results_tuned, all_results = run_model(
                model_id, task_label, class_names,
                X_test, y_test, test_mapped, dirs,
                all_classwise, mean_results, mean_results_tuned, all_results,
                n_trials=n_trials
            )
        except Exception as e:
            print(f"    ERROR [{model_id}]: {e}")

    # ── Save summary files ─────────────────────────────────────────────────────
    pd.concat(mean_results,       axis=1).T.to_csv(
        f"{dirs['results']}/combined_results_original.csv", index=False)
    pd.concat(mean_results_tuned, axis=1).T.to_csv(
        f"{dirs['results']}/combined_results_tuned.csv",    index=False)
    all_results.to_csv(
        f"{dirs['results']}/all_results.csv", index=False)
    all_classwise.reset_index(drop=True).to_csv(
        f"{dirs['results']}/all_classwise_metrics.csv", index=False)

    print(f"\n  ✓ {task_name} outputs saved to:")
    print(f"    results/{task_label}/")
    print(f"    models/{task_label}/")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Cancer classification pipeline")
    parser.add_argument(
        '--task',
        choices=['multiclass', 'three_class', 'binary', 'all'],
        default='all',
        help="Which classification task to run (default: all)"
    )
    parser.add_argument(
        '--optuna_itr',
        type=int,
        default=10,
        help="Number of Optuna iterations for hyperparameter tuning (default: 10)"
    )
    args = parser.parse_args()

    tasks = list(TASK_CONFIGS.keys()) if args.task == 'all' else [args.task]

    for task in tasks:
        run_task(task, n_trials=args.optuna_itr)

    print("\n✓ Pipeline complete.")


if __name__ == "__main__":
    main()
