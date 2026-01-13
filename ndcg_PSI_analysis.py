import os
import json
import pandas as pd
import numpy as np
from scipy.stats import linregress, pearsonr, spearmanr, kendalltau
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="all", help="Name of the model to analyze")
args = parser.parse_args()

all_info = {}
json_files = [f"evaluation_results/{args.model_name}/{args.model_name}.json",]
# json_files = [
#     "evaluation_results/bge-m3/bge-m3.json",
#     "evaluation_results/gte-multilingual-base/gte-multilingual-base.json",
#     "evaluation_results/inf-retriever-v1-1.5b/inf-retriever-v1-1.5b.json",
#     "evaluation_results/inf-retriever-v1/inf-retriever-v1.json",
#     "evaluation_results/KaLM-Embedding-Gemma3-12B-2511/KaLM-Embedding-Gemma3-12B-2511.json",
#     "evaluation_results/NV-Embed-v2/NV-Embed-v2.json",
#     "evaluation_results/llama-embed-nemotron-8b/llama-embed-nemotron-8b.json",
#     "evaluation_results/Qwen3-Embedding-0.6B/Qwen3-Embedding-0.6B.json",
#     "evaluation_results/Qwen3-Embedding-4B/Qwen3-Embedding-4B.json",
#     "evaluation_results/Qwen3-Embedding-8B/Qwen3-Embedding-8B.json",
# ]

for json_file in tqdm(json_files):
    with open(json_file, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    model_name = json_file.split("/")[-1].replace(".json","")
    all_info[model_name]={}
    for language_setting in json_data:
        all_info[model_name][language_setting] = {}
        all_items = []
        for domain in json_data[language_setting]:
            d = json_data[language_setting][domain]
            for k in d["evaluation_results"].keys():
                v = d["evaluation_results"][k]
                # print(v.keys())
                all_items.append({
                        "position": float(sum(v["pos_char_span"])/2),
                        "total_length": v["pos_char_length"],
                        "token_length": v["pos_token_length"],
                        "score": v["scores"]["ndcg_cut_10"]
                })

        df = pd.DataFrame(all_items, columns=["position", "total_length", "token_length", "score"])
        df = df[(df["total_length"] > 0) & (df["position"] >= 0)]
        df["position"] = np.minimum(df["position"], df["total_length"])
        df["relpos"] = (df["position"] / df["total_length"]).clip(0,1)
        df["log_len"] = np.log(df["total_length"] + 1)
        # df["len_quartile"] = pd.qcut(df["total_length"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
        bins = [0, 512, 1024, 1536, np.inf]
        labels = ["Q1", "Q2", "Q3", "Q4"]
        df["len_quartile"] = pd.cut(
            df["token_length"],
            bins=bins,
            labels=labels,
            right=True
        )
        # print(df["len_quartile"].value_counts())

        bins = np.linspace(0, 1, 21)
        df["relpos_bin"] = pd.cut(df["relpos"], bins=bins, include_lowest=True)

        binned = df.groupby("relpos_bin", observed=True)["score"].agg(["mean", "count"]).reset_index()
        binned_q = (
            df.groupby(["len_quartile", "relpos_bin"], observed=True)["score"]
            .agg(["mean", "count"])
            .reset_index()
        )
        
        def interval_center(iv):
            import pandas as pd, numpy as np
            if pd.isna(iv):
                return np.nan
            return float(iv.left) + (float(iv.right) - float(iv.left)) / 2.0

        binned["bin_center"] = binned["relpos_bin"].apply(interval_center)
        binned_q["bin_center"] = binned_q["relpos_bin"].apply(interval_center)

        results = {}

        for q in ["Q1", "Q2", "Q3", "Q4"]:
            sub = binned_q[binned_q["len_quartile"] == q].sort_values("bin_center")
            x = sub["bin_center"].values
            y = sub["mean"].values
            
            # 线性回归
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            
            # Pearson
            pearson_r, pearson_p = pearsonr(x, y)
            
            # Spearman
            spearman_rho, spearman_p = spearmanr(x, y)
            
            # Kendall
            kendall_tau, kendall_p = kendalltau(x, y)
            
            results[q] = {
                "nDCG@10": float(np.mean(y)),
                "PSI": float(1 - min(y)/max(y)),
                "slope": slope,
                "intercept": intercept,
                "r_squared": r_value**2,
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_rho": spearman_rho,
                "spearman_p": spearman_p,
                "kendall_tau": kendall_tau,
                "kendall_p": kendall_p
            }
        
        mask = binned["mean"].notna() & binned["bin_center"].notna()
        x_b = binned.loc[mask, "bin_center"].values.reshape(-1,1)
        y_b = binned.loc[mask, "mean"].values
        all_info[model_name][language_setting] = {
            "nDCG@10": float(np.mean([s["score"] for s in all_items])),
            "nDCG@10(Q1)": float(results["Q1"]["nDCG@10"]),
            "nDCG@10(Q2)": float(results["Q2"]["nDCG@10"]),
            "nDCG@10(Q3)": float(results["Q3"]["nDCG@10"]),
            "nDCG@10(Q4)": float(results["Q4"]["nDCG@10"]),

            "PSI": float(1 - min(y_b)/max(y_b)),
            "PSI(Q1)": float(results["Q1"]["PSI"]),
            "PSI(Q2)": float(results["Q2"]["PSI"]),
            "PSI(Q3)": float(results["Q3"]["PSI"]),
            "PSI(Q4)": float(results["Q4"]["PSI"]),
        }

with open(f"evaluation_results/{args.model_name}_eval_metrics.json", "w") as f:
    f.write(json.dumps(all_info, ensure_ascii=False, indent=4))