from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

output_dir = Path("data") / "oof_v2"
output_dir.mkdir(parents=True, exist_ok=True)

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
# LightGBM ハイパーパラメータ（最小限）
lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.03,
    "n_estimators": 50000,  # early_stopping ありきで多めに
    "num_leaves": 16,
    "max_depth": 5,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "random_state": 0,
}


for i, csv_files in enumerate(csv_list):
    print(f"csv_files: {csv_files}")
    # Load CSV files
    train_csv_list, val_csv_list, test_csv_list = [], [], []
    for j, csv_file in enumerate(csv_list):
        if i == j:
            test_csv_list.append(csv_file)
        else:
            train_csv_list.append(csv_file)
    print(i + 1 % len(csv_list), len(csv_list))
    print(f"train_csv_list: {train_csv_list}")
    print(f"val_csv_list: {val_csv_list}")
    print(f"test_csv_list: {test_csv_list}")
    train_sample_id_list, train_true_mos_list, train_pred_mos_list = load_csv_list(csv_list)
    test_sample_id_list, test_true_mos_list, test_pred_mos_list = load_csv_list(test_csv_list)
    print(f"train_true_mos_list.shape: {train_true_mos_list.shape}")
    print(f"train_pred_mos_list.shape: {train_pred_mos_list.shape}")
    assert train_pred_mos_list.shape[0] > 0, f"train_pred_mos_list is empty for {csv_files}"
    assert test_pred_mos_list.shape[0] > 0, f"test_pred_mos_list is empty for {csv_files}"
    # ----------------------------
    model = LGBMRegressor(**lgb_params)
    model.fit(
        train_pred_mos_list,
        train_true_mos_list,
        eval_set=[(test_pred_mos_list, test_true_mos_list)],
        eval_metric="rmse",
        callbacks=[
            lgb.early_stopping(stopping_rounds=500, verbose=True),
            lgb.log_evaluation(period=100),
        ],
    )

    pred_mos_list = model.predict(test_pred_mos_list, num_iteration=model.best_iteration_)
    pred_mos_list = pred_mos_list * 2.0 + 3.0  # pyright: ignore[reportOperatorIssue]
    test_true_mos_list = test_true_mos_list * 2.0 + 3.0
    output_list = []
    for sample_id, true_mos, pred_mos in zip(test_sample_id_list, test_true_mos_list, pred_mos_list, strict=False):
        output_list.append(
            {
                "sample_id": sample_id,
                "true_mos": true_mos,
                "pred_mos": pred_mos,
            }
        )
    output_df = pd.DataFrame(output_list)
    output_csv = output_dir / f"oof_{Path(csv_files[0]).stem}.csv"
    output_df.to_csv(output_csv, index=False)
    print(f"Output saved to {output_csv}")
