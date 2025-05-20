from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor  # pip install xgboost

csv_dir_list = [
    "data/i3000",
    "data/i4000",
    "data/i5000",
]
csv_list = []
tmp_csv_list = sorted(list(Path(csv_dir_list[0]).glob("*.csv")))
for csv_file in tmp_csv_list:
    fold_csv_list = [Path(csv_dir) / csv_file.name for csv_dir in csv_dir_list]
    fold_csv_list = sorted(fold_csv_list)
    csv_list.append(fold_csv_list)
print(f"csv_list: {csv_list}")


def load_csv_list(csv_list: list) -> tuple:
    """Load CSV files into a list of dataframes."""
    sample_id_list, true_mos_list, pred_mos_dict = [], [], {}
    for csv_files in csv_list:
        for i, csv_file in enumerate(csv_files):
            data_df = pd.read_csv(csv_file)
            sample_id = data_df["sample_id"].tolist()
            true_mos = data_df["true_mos"].tolist()
            true_mos = [(x - 3) / 2.0 for x in true_mos]  # 1-5 -> 1-7
            pred_mos = data_df["pred_mos"].tolist()
            pred_mos = [(x - 3) / 2.0 for x in pred_mos]  # 1-5 -> 1-7
            if i == 0:
                sample_id_list.extend(sample_id)
                true_mos_list.extend(true_mos)
            if i not in pred_mos_dict:
                pred_mos_dict[i] = []
            pred_mos_dict[i].extend(pred_mos)
    pred_mos_list = [pred_mos_dict[i] for i in range(len(pred_mos_dict))]
    pred_mos_list = np.array(pred_mos_list).T
    true_mos_list = np.array(true_mos_list)
    return sample_id_list, true_mos_list, pred_mos_list


# ----------------------------
# 1. メタモデル候補を準備
# ----------------------------
meta_models = [
    ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 13))),
    ("lasso", LassoCV(alphas=np.logspace(-3, 1, 20), max_iter=5000)),
    ("elastic", ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 1], alphas=np.logspace(-3, 1, 20), max_iter=5000)),  # pyright: ignore[reportArgumentType]
    ("gbr", GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, random_state=0)),
    ("xgb", XGBRegressor(n_estimators=400, learning_rate=0.05, objective="reg:squarederror", random_state=0)),
    ("lgbm", LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=0)),
]

metric = make_scorer(mean_squared_error, greater_is_better=False)  # neg-RMSE
cv_outer = KFold(n_splits=10, shuffle=True, random_state=42)
results = {}

train_sample_id_list, train_true_mos_list, train_pred_mos_list = load_csv_list(csv_list)
print(train_pred_mos_list.shape)
print(train_true_mos_list.shape)
for name, model in meta_models:
    scores = cross_val_score(model, train_pred_mos_list, train_true_mos_list, scoring=metric, cv=cv_outer, n_jobs=-1)
    results[name] = scores
    print(f"{name:7s}  CV RMSE (mean±std) = {-scores.mean():.5f} ± {scores.std():.5f}")


best_name = min(results, key=lambda n: -results[n].mean())  # mean が最も大きい（neg RMSE 最小）
best_model = dict(meta_models)[best_name]
print(f"\n>>> Selected meta-model: {best_name} " f"(CV RMSE = {-results[best_name].mean():.5f})")
