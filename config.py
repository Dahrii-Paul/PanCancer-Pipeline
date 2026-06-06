# ── PyCaret setup config (shared) ─────────────────────────────────────────────
PYCARET_CONFIG = {
    'session_id':      123,
    'fold':            5,
    'fold_strategy':   'stratifiedkfold',
    'imputation_type': None,
    'n_jobs':          -1
}

# ── Models to train (shared) ──────────────────────────────────────────────────
MODELS_LIST = ['rf', 'et', 'xgboost']

BASE_OUTPUT_DIR = "./models"

# ── Task-specific configs (each has its own data files) ───────────────────────
TASK_CONFIGS = {

    'multiclass': {
        'train_path':     './data/multiclass_train.csv',
        'test_path':      './data/multiclass_test.csv',
        'target_col':     'Class',
        'label':          'multiclass',
        'target_mapping': {
            'BLCA': 0,  'BRCA': 1,  'CESC': 2,  'CHOL': 3,  'COAD': 4,
            'ESCA': 5,  'GBM': 6,   'HNSC': 7,  'KICH': 8,  'KIRC': 9,
            'KIRP': 10, 'LIHC': 11, 'LUAD': 12, 'LUSC': 13, 'Normal': 14,
            'PAAD': 15, 'PCPG': 16, 'PRAD': 17, 'READ': 18, 'SARC': 19,
            'SKCM': 20, 'STAD': 21, 'THCA': 22, 'THYM': 23, 'UCEC': 24
        }
    },

    'three_class': {
        'train_path':     './data/three_class_train.csv',
        'test_path':      './data/three_class_test.csv',
        'target_col':     'LC_labels',
        'label':          'three_class',
        'target_mapping': {
            'LIHC': 0, 'Others': 1, 'Normal': 2
        }
    },

    'binary': {
        'train_path':     './data/binary_train.csv',
        'test_path':      './data/binary_test.csv',
        'target_col':     'LC_labels',
        'label':          'binary',
        'target_mapping': {
            'LIHC': 0, 'Others': 1
        }
    }
}
