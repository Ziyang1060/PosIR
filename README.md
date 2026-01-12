# PosIR: Position-Aware Heterogeneous Information Retrieval Benchmark

> A large-scale heterogeneous benchmark for diagnosing position bias in retrieval models.

# Overview

## 🔑 Key Features

- 🎯 **Position-aware relevance** with span-level grounding
- 📏 **Disentangles** document length from evidence position
- 🌍 **310 datasets** · **10 languages** · **31 domains**
- 🔍 **Diagnoses position bias** (primacy and recency) in information retrieval
- 🧪 **Supports gradient-based saliency analysis** for investigating internal mechanisms

## Resources

- Dataset: https://huggingface.co/datasets/infgrad/PosIR-Benchmark-v1
- Paper: Coming soon (arXiv preprint in preparation)
- Leaderboard: Coming soon — contributions welcome!



# Abstract
While dense retrieval models have achieved remarkable success, rigorous evaluation of their sensitivity to the position of relevant information (i.e., position bias) remains largely unexplored. Existing benchmarks typically employ position-agnostic relevance labels, conflating the challenge of processing long contexts with the bias against specific evidence locations. To address this challenge, we introduce PosIR (Position-Aware Information Retrieval), a comprehensive benchmark designed to diagnose position bias in diverse retrieval scenarios. PosIR comprises 310 datasets spanning 10 languages and 31 domains, constructed through a rigorous pipeline that ties relevance to precise reference spans, enabling the strict disentanglement of document length from information position. Extensive experiments with 10 state-of-the-art embedding models reveal that: (1) Performance on PosIR in long-context settings correlates poorly with the MMTEB benchmark, exposing limitations in current short-text benchmarks; (2) Position bias is pervasive and intensifies with document length, with most models exhibiting primacy bias while certain models show unexpected recency bias; (3) Gradient-based saliency analysis further uncovers the distinct internal attention mechanisms driving these positional preferences. In summary, PosIR serves as a valuable diagnostic framework to foster the development of position-robust retrieval systems.

# Usage

## Installation
We recommend managing the environment with `uv` and Python 3.12. Different retrieval models may require specific library versions; see the notes below for NV-Embed-v2.

```sh
pip install uv
uv venv posir --python 3.12 --seed
source posir/bin/activate
uv pip install polars
uv pip install transformers
uv pip install sentence_transformers
uv pip install scikit-learn
uv pip install pandas
uv pip install pytrec_eval
uv pip install psutil
uv pip install --no-build-isolation flash-attn
uv pip install datasets
uv pip install einops

# For nvidia/NV-Embed-v2
# uv pip install transformers==4.45.1
# uv pip install sentence_transformers==3.2.1
```

## Evaluation Pipeline
1) Download the dataset into `PosIR-Benchmark-v1/`.
2) In `eval.sh`, set `target_query_allowed_langs`, `target_corpus_language`, and `MODEL_PATH` to run monolingual or cross-lingual retrieval. Then run:
    ```sh
    bash eval.sh
    ```
    Detailed evaluation results for each domain will be written to a new directory under `evaluation_results/model_name/language_mode/`, e.g., `evaluation_results/Qwen3-Embedding-8B/fra-eng/accommodation_catering_hotel.json`. "fra-eng" means the retrieval is performed in the French-English language mode.

3) Aggregate results for 31 domains:
    ```sh
    python agg_result.py --model_name Qwen3-Embedding-8B
    ```
    Aggregated results are saved as JSON, e.g., `evaluation_results/Qwen3-Embedding-8B/Qwen3-Embedding-8B.json`.
4) Compute NDCG and PSI metrics (all models):
    ```sh
    python ndcg_PSI_analysis.py
    ```
    Macro-weighted NDCG and PSI metrics across 31 domains are stored in `evaluation_results/all_eval_metrics.json`.
5) (Optional) Visualize results (all models):
    ```sh
    python draw_fig.py
    ```
    Figures are saved under `figs/` (examples are provided).

## Gradient-based Saliency Analysis
We provide experiment scripts for `Qwen3-Embedding-8B` and `NV-Embed-v2`. 
> For NV-Embed-v2, replace the original `modeling_nvembed.py` in the model directory with the version in this repository.

```sh
python gradient_saliency/qwen3_exp.py
python gradient_saliency/nvidia_exp.py
```

Results are persisted as `.pkl` files and can be visualized with:
```sh
python gradient_saliency/visualize.py
```