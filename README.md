# PosIR: A Multilingual Benchmark for Diagnosing Position Bias in Information Retrieval

> An information retrieval benchmark designed to diagnose position bias across multilingual and cross-lingual retrieval.

# Overview

## 🔑 Key Features

- 🎯 **Position-aware relevance** with span-level grounding
- 📏 **Disentangles** document length from evidence position
- 🌍 **310 datasets** · **10 languages** · **31 domains**
- 🔍 **Diagnoses position bias** (primacy and recency) in information retrieval
- 🧪 **Supports gradient-based saliency analysis** for investigating internal mechanisms

## Resources

- Dataset: https://huggingface.co/datasets/infgrad/PosIR-Benchmark-v1
- Supplementary material: [Supplementary Material](docs/Supplementary_Material.pdf)
- Arxiv Paper: https://arxiv.org/abs/2601.08363



# Abstract
Position bias—the systematic effect of where key information is located on retrieval effectiveness—is a critical concern in information retrieval. Although recent studies have identified such bias in dense retrievers, existing analyses focus predominantly on monolingual English, fail to disentangle document length from information position, and lack a standardized framework for controlled evaluation. To address these limitations, we introduce PosIR, a benchmark to systematically diagnose position bias across diverse retrieval scenarios, comprising 310 language–domain dataset combinations spanning 10 languages and 31 domains. Each query is paired with a positive document, a localized reference span, and positional metadata, enabling controlled analysis of retrieval quality across evidence locations and length bins. PosIR provides a complete data generation pipeline with quality-control filtering, along with baseline evaluation scripts. Experiments with 10 widely-used dense retrievers reveal that position bias is pervasive in both multilingual and cross-lingual retrieval, and that it intensifies as documents grow longer. PosIR serves as a foundational resource to advance the development of position-robust retrieval systems.

# Usage

## Installation

There are two ways to evaluate the retrieval models:

### sentence-transformers
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
uv pip install seaborn
uv pip install datasets
uv pip install einops

# For nvidia/NV-Embed-v2
# uv pip install transformers==4.45.1
# uv pip install sentence_transformers==3.2.1
```

### vLLM

**We strongly recommend use [vLLM](https://docs.vllm.ai/en/stable/getting_started/installation) to deploy the embedding models for better efficiency in evaluation.**


## Evaluation Pipeline
1) Download the dataset into `PosIR-Benchmark-v1/`.
2) In `eval.sh`, set `target_query_allowed_langs`, `target_corpus_language`, and `MODEL_PATH` to run monolingual or cross-lingual retrieval. Then run:
    ```sh
    bash eval.sh # for sentence-transformers
    bash eval_vllm.sh # for vLLM
    ```
    Detailed evaluation results for each domain will be written to a new directory under `evaluation_results/model_name/language_mode/`, e.g., `evaluation_results/Qwen3-Embedding-8B/fra-eng/accommodation_catering_hotel.json`. "fra-eng" means the retrieval is performed in the French-English language mode.

3) Aggregate results for 31 domains:
    ```sh
    python agg_result.py --model_name Qwen3-Embedding-8B
    ```
    Aggregated results are saved as JSON, e.g., `evaluation_results/Qwen3-Embedding-8B/Qwen3-Embedding-8B.json`.
4) Compute NDCG and PSI metrics:
    ```sh
    python ndcg_PSI_analysis.py --model_name Qwen3-Embedding-8B
    ```
    Macro-weighted NDCG and PSI metrics across 31 domains are stored in `evaluation_results/Qwen3-Embedding-8B_eval_metrics.json`.
5) (Optional) Visualize results for all evaluated models:
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