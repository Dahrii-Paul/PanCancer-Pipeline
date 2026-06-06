# Cancer Classification Pipeline

Multi-class, three-class, and binary classification of cancer types using RNA-seq gene expression data (TCGA Pan-Cancer dataset). All three tasks run from a single unified pipeline.

## Tasks

| Task | Classes | Target Column |
|---|---|---|
| `multiclass` | 24 TCGA cancer types + Normal | `Class` |
| `three_class` | LIHC / Others / Normal | `LC_labels` |
| `binary` | LIHC / Others | `LC_labels` |

## Models
Random Forest, Extra Trees, XGBoost — each trained as original and Optuna-tuned.

## Structure
```
cancer-classification/
├── data/
│   ├── train.csv  
│   └── test.csv 
├── models/
│   ├── multiclass/
│   │   ├── saved_models/
│   │   ├── confusion_matrices/
│   │   └── classwise_metrics/
│   ├── three_class/
│   └── binary/
├── results/
│   ├── multiclass/
│   ├── three_class/
│   └── binary/
├── config.py 			# paths, target mappings, task configs
├── evaluate.py			# metric functions
├── train.py 			# unified pipeline
├── requirements.txt
└── README.md
```
---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `pycaret[full]` | 3.3.2 | ML pipeline (training, tuning, evaluation) |
| `scikit-learn` | 1.5.2 | Metrics — confusion matrix, AUC, MCC |
| `pandas` | 2.2.3 | Data loading and results saving |
| `numpy` | 1.26.4 | Numerical computations |
| `optuna` | 4.1.0 | Hyperparameter tuning (used inside PyCaret) |
| `joblib` | 1.4.2 | Model serialization |

> **Note:** `pycaret[full]` installs most of the above automatically.  
> Pinned versions ensure reproducibility across machines.

---
## Setup

```bash
# You can place your data files and run
cp your_train.csv data/train.csv
cp your_test.csv  data/test.csv
```

## Usage

```bash
# Run a specific task
python train.py --task multiclass
python train.py --task three_class
python train.py --task binary

# Run all three tasks in sequence
python train.py --task all
```

## Outputs (per task)
| File | Description |
|---|---|
| `results/<task>/combined_results_original.csv` | CV metrics — original models |
| `results/<task>/combined_results_tuned.csv` | CV metrics — tuned models |
| `results/<task>/all_results.csv` | Train + test results combined |
| `results/<task>/all_classwise_metrics.csv` | Per-class Sensitivity, Specificity, Precision, F1, MCC, AUC |
| `models/<task>/confusion_matrices/` | One CSV per model variant |
| `models/<task>/saved_models/` | `.pkl` files (original + tuned) |
