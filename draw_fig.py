import os
import json
import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")
from PIL import Image

# mode = "eng-eng"
mode = "fra-eng"
dir_paths = [
    "evaluation_results/bge-m3/" + mode,
    "evaluation_results/gte-multilingual-base/" + mode,
    "evaluation_results/inf-retriever-v1/" + mode,
    "evaluation_results/inf-retriever-v1-1.5b/" + mode,
    "evaluation_results/KaLM-Embedding-Gemma3-12B-2511/" + mode,
    "evaluation_results/llama-embed-nemotron-8b/" + mode,
    "evaluation_results/NV-Embed-v2/" + mode,
    "evaluation_results/Qwen3-Embedding-0.6B/" + mode,
    "evaluation_results/Qwen3-Embedding-4B/" + mode,
    "evaluation_results/Qwen3-Embedding-8B/" + mode
]
folder_path = f'figs/{mode}_figs'
os.makedirs(folder_path, exist_ok=True)

for dir_path in tqdm.tqdm(dir_paths):
    model_name = dir_path.split("/")[-2]

    data = {}
    all_items = []
    for filename in os.listdir(dir_path):
        if filename.endswith(".json"):
            file_path = os.path.join(dir_path, filename)
            
            with open(file_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
            
            eval_res = json_data.get("evaluation_results", {})
            key = os.path.splitext(filename)[0]
            data[key] = eval_res
            for k in eval_res:
                v = eval_res[k]
                # print(v.keys())
                all_items.append({
                    "position": float(sum(v["pos_char_span"])/2),
                    "total_length": v["pos_char_length"],
                    "token_length": v["pos_token_length"],
                    "score": v["scores"]["ndcg_cut_10"]
                })

    import pandas as pd
    import numpy as np

    df = pd.DataFrame(all_items, columns=["position", "total_length","token_length", "score"])
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

    bins = np.linspace(0, 1, 21)
    df["relpos_bin"] = pd.cut(df["relpos"], bins=bins, include_lowest=True)

    binned = df.groupby("relpos_bin")["score"].agg(["mean", "count"]).reset_index()
    binned_q = (
        df.groupby(["len_quartile", "relpos_bin"])["score"]
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


    fig2 = plt.figure(figsize=(6,4), dpi=140)
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        sub = binned_q[binned_q["len_quartile"] == q].sort_values("bin_center")
        plt.plot(sub["bin_center"], sub["mean"], marker="o", label=q)
    plt.xlabel("Relative Position", fontsize=22)
    plt.ylabel("Mean nDCG@10", fontsize=20)
    model_name_1 = model_name
    if model_name == "KaLM-Embedding-Gemma3-12B-2511":
        model_name_1 = "KaLM-Embedding-12B"
    plt.title(f"{model_name_1}", fontsize=24)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"figs/{mode}_figs/{model_name}.png")

# image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
image_files = [
    folder_path + "/gte-multilingual-base.png",
    folder_path + "/bge-m3.png",
    folder_path + "/Qwen3-Embedding-0.6B.png",
    folder_path + "/inf-retriever-v1-1.5b.png",
    folder_path + "/Qwen3-Embedding-4B.png",
    folder_path + "/inf-retriever-v1.png",
    folder_path + "/NV-Embed-v2.png",
    folder_path + "/llama-embed-nemotron-8b.png",
    folder_path + "/Qwen3-Embedding-8B.png",
    folder_path + "/KaLM-Embedding-Gemma3-12B-2511.png",
]

images = [Image.open(f) for f in image_files]
image_width, image_height = images[0].size
num_images = len(images)
rows = 2
cols = 5
total_width = cols * image_width
total_height = rows * image_height

combined_image = Image.new('RGB', (total_width, total_height))

for i, image in enumerate(images):
    row = i // cols
    col = i % cols
    x_offset = col * image_width
    y_offset = row * image_height
    combined_image.paste(image, (x_offset, y_offset))

combined_image.save(f'figs/combined_{mode}_figs.png')
