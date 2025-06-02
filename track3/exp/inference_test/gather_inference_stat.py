from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

# python ./track3/exp/inference_test/gather_inference_stat.py --input_dir_list  ./data/i4000 --output_txt ./data/track3/i4000.txt


def calculate_mean(values: list) -> float:
    """Calculate the mean of a list of values."""
    with pm.Model() as _:
        # (1) 平均 μ の事前分布（弱情報的に広めに取っておく）
        mu = pm.Normal("mu", mu=3, sigma=50)
        # (2) スケール σ の事前（HalfNormal によって正の値のみ）
        sigma = pm.HalfNormal("sigma", sigma=5)
        # (3) 自由度 ν の事前（Exponential）
        #     lam=1/10 とすることで E[ν]=10 前後の想定
        nu = pm.Exponential("nu", lam=1 / 10)
        # (4) 尤度に StudentT 分布を指定
        #     observed=data とすると data の各要素が
        #       x_i ~ StudentT(nu, mu, sigma)
        #     から生成された、というモデルになる
        y_obs = pm.StudentT("y_obs", nu=nu, mu=mu, sigma=sigma, observed=np.array(values))
        _ = y_obs
        # ── 3) サンプリング設定 ──────────────────
        trace = pm.sample(
            draws=2000,  # 事後サンプル数
            tune=1000,  # バーンイン（ウォームアップ）ステップ数
            target_accept=0.9,  # NUTS の受理率目標を高くする
            return_inferencedata=True,
            cores=1,  # 並列数（環境に応じて変更可）
            random_seed=42,
        )
        # ── 4) 事後分布の要約を表示 ──────────────────
        print(az.summary(trace, var_names=["mu", "sigma", "nu"], hdi_prob=0.95))
        return trace.posterior["mu"].to_numpy().mean()  # pyright: ignore[reportAttributeAccessIssue]


def gather_inference(
    input_dir_list: str,
    output_txt: str,
) -> None:
    """Gather inference results from multiple CSV files into a single CSV file."""
    # Initialize an empty list to store the dataframes
    # Iterate through all CSV files in the input directory
    sample_id_dict = {}
    for input_dir in input_dir_list:
        for csv_file in Path(input_dir).glob("*.csv"):
            # Read the CSV file into a dataframe
            data_df = pd.read_csv(csv_file)

            tmp_sample_id_list = data_df["sample_id"].tolist()
            tmp_pred_mos_list = data_df["pred_mos"].tolist()

            for sample_id, pred_mos in zip(tmp_sample_id_list, tmp_pred_mos_list, strict=False):
                if sample_id not in sample_id_dict:
                    sample_id_dict[sample_id] = []
                sample_id_dict[sample_id].append(pred_mos)

    with Path(output_txt).open("w") as f:
        for sample_id, pred_mos_list in sample_id_dict.items():
            pred_mos = calculate_mean(pred_mos_list)
            f.write(f"{sample_id},{pred_mos}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gather inference results")
    parser.add_argument(
        "--input_dir_list", type=str, nargs="+", required=True, help="List of directories containing the CSV files"
    )
    parser.add_argument("--output_txt", type=str, required=True, help="Path to the output text file")
    args = parser.parse_args()
    input_dir_list = args.input_dir_list
    output_txt = args.output_txt
    Path(output_txt).parent.mkdir(parents=True, exist_ok=True)
    # Run the gather_inference function
    gather_inference(input_dir_list, output_txt)
