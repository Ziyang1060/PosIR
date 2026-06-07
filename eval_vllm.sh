#!/bin/bash
DATASET="PosIR-Benchmark-v1"

### Main Languages：("ara-Arab"  "cmn-Hans" "deu-Latn" "eng-Latn" "fra-Latn" "ita-Latn" "kor-Kore" "por-Latn" "rus-Cyrl" "spa-Latn")
### More Languages: ("hin-Deva" "pol-Latn" "ben-Beng" "jpn-Jpan")

# Monolingual Retrieval
target_query_allowed_langs=("eng-Latn" "cmn-Hans")
target_corpus_languae="" ### Corpus Language: default value is "", meaning it will keep the same language as the query.

# Cross-lingual Retrieval (e.g. Fra -> Eng)
# target_query_allowed_langs=("fra-Latn")
# target_corpus_languae="eng-Latn"


MODEL_PATH=(
    "your_path/Qwen/Qwen3-Embedding-8B"
    "your_path/Qwen/Qwen3-Embedding-4B"
    "your_path/Qwen/Qwen3-Embedding-0.6B"
    )

for model in "${MODEL_PATH[@]}"; do
    echo "================ MODEL: $model ================"

    # --- 启动 vLLM embedding 服务 ---
    VLLM_LOG="vllm_$(basename "$model")_$(date +%Y%m%d_%H%M%S).log"
    vllm serve "$model" \
        --runner pooling \
        --data-parallel-size 8 \ # Data-parallel size MEANS the number of GPUs.
        --host 0.0.0.0 --port 8000 \
        > "$VLLM_LOG" 2>&1 &
    VLLM_PID=$!
    echo "vLLM PID: $VLLM_PID  log: $VLLM_LOG"

    # 等待就绪
    echo "Waiting for vLLM to be ready..."
    while ! curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; do
        sleep 5
    done
    echo "vLLM ready."

    for lang in "$DATASET"/*/; do
        [ -d "$lang" ] || continue
        lang=${lang%/}

        lang_name=$(basename "$lang")
        if [[ ! " ${target_query_allowed_langs[@]} " =~ " $lang_name " ]]; then
            continue
        fi

        start_time=$(date +%s)
        for domain in "$lang"/*/; do
            [ -d "$domain" ] || continue

            domain_name=$(basename "$domain")

            queries_path="${domain%/}/queries.parquet"
            qrels_path="${domain%/}/qrels/test.parquet"
            corpus_path="${domain%/}/corpus.parquet"

            if [[ -n "$target_corpus_languae" ]]; then
                corpus_path="${corpus_path/$lang_name/$target_corpus_languae}"
            fi

            if [ ! -f "$queries_path" ] || [ ! -f "$corpus_path" ] || [ ! -f "$qrels_path" ]; then
                echo "Warning: Skipping $lang_name / $domain_name (missing files)"
                continue
            fi

            echo "Running: $lang_name / $domain_name"
            python eval_vllm.py \
                --queries_path "$queries_path" \
                --corpus_path "$corpus_path" \
                --qrels_path "$qrels_path" \
                --model_path "$model" \
                --batch_size 512 # Batch size for evaluation.
        done

        end_time=$(date +%s)
        elapsed=$((end_time - start_time))

        echo "----------------------------------------\n"
        printf "Milestone: Running: %s Total time: %dh %dm %ds\n" \
            "$lang_name" "$((elapsed/3600))" "$(((elapsed%3600)/60))" "$((elapsed%60))"
        echo "----------------------------------------\n"
    done

    # --- 彻底停掉 vLLM ---
    echo "Stopping vLLM (PID $VLLM_PID)..."
    # 先杀所有子进程，再杀父进程，最后确保端口释放
    pkill -TERM -P $VLLM_PID 2>/dev/null
    sleep 2
    kill -TERM $VLLM_PID 2>/dev/null
    sleep 2
    # 如果还有残留，强制清理
    pkill -KILL -P $VLLM_PID 2>/dev/null
    kill -KILL $VLLM_PID 2>/dev/null
    # 兜底：杀掉一切占用 8000 端口的进程
    fuser -k 8000/tcp 2>/dev/null
    echo "================ MODEL: $model done ================"
done