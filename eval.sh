#!/bin/bash
DATASET="PosIR-Benchmark-v1"

### Main Languages：("ara-Arab"  "cmn-Hans" "deu-Latn" "eng-Latn" "fra-Latn" "ita-Latn" "kor-Kore" "por-Latn" "rus-Cyrl" "spa-Latn")
### More Languages: ("hin-Deva" "pol-Latn" "ben-Beng" "jpn-Jpan")

# Monolingual Retrieval
target_query_allowed_langs=("eng-Latn")
target_corpus_languae="" ### Corpus Language: default value is "", meaning it will keep the same language as the query.

# Cross-lingual Retrieval (e.g. Fra -> Eng)
# target_query_allowed_langs=("fra-Latn")
# target_corpus_languae="eng-Latn"


MODEL_PATH=(
    "Qwen/Qwen3-Embedding-8B"
    "Qwen/Qwen3-Embedding-4B"
    "Qwen/Qwen3-Embedding-0.6B"
    )

for model in "${MODEL_PATH[@]}"; do
    echo "================ MODEL: $model ================"
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
            python eval.py \
                --queries_path "$queries_path" \
                --corpus_path "$corpus_path" \
                --qrels_path "$qrels_path" \
                --model_path "$model" \
                --batch_size 64
        done
        
        end_time=$(date +%s)
        elapsed=$((end_time - start_time))

        echo "----------------------------------------\n"
        printf "Milestone: Running: %s Total time: %dh %dm %ds\n" \
            "$lang_name" "$((elapsed/3600))" "$(((elapsed%3600)/60))" "$((elapsed%60))"
        echo "----------------------------------------\n"
    done
    echo "================ MODEL: $model ================"
done


