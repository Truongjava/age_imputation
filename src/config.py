"""Cau hinh pipeline age imputation (H&M dataset)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CUSTOMERS_CSV = ROOT / "customers.csv"
ARTICLES_CSV = ROOT / "articles.csv"
TXN_CSV = ROOT / "transactions_train.csv"

ARTIFACTS = ROOT / "artifacts"
OUTPUTS = ROOT / "outputs"

TXN_PARQUET = ARTIFACTS / "transactions_compact.parquet"
CUSTOMER_FEATURES_PARQUET = ARTIFACTS / "customer_features.parquet"
CATEGORY_MAPS_JSON = ARTIFACTS / "category_maps.json"
FEATURE_META_JSON = ARTIFACTS / "feature_meta.json"
METRICS_JSON = OUTPUTS / "metrics.json"
IMPUTED_CSV = OUTPUTS / "customers_age_imputed.csv"
REPORT_MD = OUTPUTS / "age_imputation_report.md"

AGE_MIN, AGE_MAX = 16, 99

N_SPLITS = 5
RANDOM_STATE = 42
TRAIN_SAMPLE = 700_000

# Cac cot categorical cua articles duoc dung lam feature hanh vi
CAT_COLS = [
    "index_group_name",
    "index_name",
    "garment_group_name",
    "section_name",
    "product_type_name",
]

# So luong category pho bien nhat giu lai cho tung cot
TOP_K = {
    "index_group_name": 5,
    "index_name": 12,
    "garment_group_name": 15,
    "section_name": 15,
    "product_type_name": 20,
}

TARGET_ENCODING_COLS = ["te_postal_code"] + ["te_" + c for c in CAT_COLS]

LGB_PARAMS = {
    "objective": "regression_l1",
    "metric": "mae",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_data_in_leaf": 300,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "max_bin": 127,
    "verbosity": -1,
    "seed": RANDOM_STATE,
}

NUM_BOOST_ROUND = 4000
EARLY_STOPPING = 150

AGE_BINS = [16, 20, 30, 40, 50, 60, 70, 100]
AGE_BIN_LABELS = ["16-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
