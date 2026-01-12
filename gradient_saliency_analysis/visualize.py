import pickle
import pandas as pd
import seaborn as sns
sns.set_theme(style="whitegrid")
import matplotlib.pyplot as plt

with open(f"gradient_saliency/NV-Embed-v2.pkl", "rb") as f:
    nv_data = pickle.load(f)

with open(f"gradient_saliency/Qwen3-Embedding-8B.pkl", "rb") as f:
    qwen_data = pickle.load(f)


df_nv = pd.DataFrame(nv_data).melt(
    var_name="relative position",
    value_name="value"
)
df_nv["model"] = "NV-Embed-v2"

df_qwen = pd.DataFrame(qwen_data).melt(
    var_name="relative position",
    value_name="value"
)
df_qwen["model"] = "Qwen3-Embedding-8B"
df = pd.concat([df_nv, df_qwen], ignore_index=True)

plt.figure(figsize=(8, 5))
# print(f'{df["relative position"].max()=}')
df["relative position"] /= df["relative position"].max()

ax=sns.lineplot(
    data=df,
    x="relative position",
    y="value",
    hue="model",
    style="model",
    errorbar="sd"
)

ax.set_xlabel("Relative Position", fontsize=19)
ax.set_ylabel("Normalized Gradient L2 Norm", fontsize=18)
plt.legend(title="Model", fontsize=17)
plt.tight_layout()
plt.savefig("gradient_saliency/NV-vs-Qwen.png")
plt.show()
