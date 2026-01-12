import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import json
import os
import polars as pl
import pickle

language_dir = "PosIR-Benchmark-v1/eng-Latn"
model_name = "Qwen/Qwen3-Embedding-8B"
pooling_type = "last"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_name):
    print(f"Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, dtype="float16").to(device)
    model.eval()
    return model, tokenizer

def mean_pooling(last_hidden_states, attention_mask):
    token_embeddings = last_hidden_states
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def last_token_pooling(last_hidden_states, attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

def get_saliency_map(model, tokenizer, query, document, pooling_method="last"):
    with torch.no_grad():
        q_inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        q_outputs = model(**q_inputs)
        
        if pooling_method == "mean":
            q_emb = mean_pooling(q_outputs.last_hidden_state, q_inputs['attention_mask'])
        else: # last
            q_emb = last_token_pooling(q_outputs.last_hidden_state, q_inputs['attention_mask'])
        
        q_emb = F.normalize(q_emb, p=2, dim=1)

    d_inputs = tokenizer(document, return_tensors="pt", truncation=True, max_length=2048).to(device)
    input_ids = d_inputs['input_ids']
    attention_mask = d_inputs['attention_mask']

    embedding_layer = model.get_input_embeddings()
    inputs_embeds = embedding_layer(input_ids).detach()
    inputs_embeds.requires_grad = True 
    inputs_embeds.retain_grad()

    d_outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    
    if pooling_method == "mean":
        d_raw_emb = mean_pooling(d_outputs.last_hidden_state, attention_mask)
    else: # last
        d_raw_emb = last_token_pooling(d_outputs.last_hidden_state, attention_mask)
    
    d_emb = F.normalize(d_raw_emb, p=2, dim=1)

    score = torch.sum(q_emb * d_emb) # Dot product of normalized vectors = Cosine Sim

    model.zero_grad()
    score.backward()

    # gradients shape: [batch_size, seq_len, hidden_dim]
    gradients = inputs_embeds.grad[0] 
    # print(gradients.shape)
    saliency_scores = torch.norm(gradients[:-1], dim=1).cpu().numpy() # remove special tokens
    
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])[:-1]

    return tokens, saliency_scores, score.item()


def resample_to_fixed_length(saliency, target_length=100):
    original_length = len(saliency)
    original_coords = np.linspace(0, 1, original_length)
    target_coords = np.linspace(0, 1, target_length)
    resampled_saliency = np.interp(target_coords, original_coords, saliency)
    return resampled_saliency

blist = []
for domain in tqdm.tqdm(os.listdir(language_dir)):
    queries_path=f"{language_dir}/{domain}/queries.parquet"
    queries = pl.read_parquet(queries_path).rows(named=True)
    corpus = pl.read_parquet(f"{language_dir}/{domain}/corpus.parquet").rows(named=True)
    qrels = pl.read_parquet(f"{language_dir}/{domain}/qrels/test.parquet").rows(named=True)

    corpus_map = {}
    for c in corpus:
        corpus_map[c["_id"]] = c
    qrels_map = {}
    for q in qrels:
        qrels_map[q["query-id"]] = q["corpus-id"]

    json_file=f"evaluation_results/Qwen3-Embedding-8B/eng-eng/{domain}.json"
    with open(json_file, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    alist = []
    for q in queries:
        pos_char_span = q["pos_char_span"]
        pos_char_length = q["pos_char_length"]
        relative_pos = (pos_char_span[0] + pos_char_span[1]) / 2 / pos_char_length
        ndcg = json_data["evaluation_results"][q["_id"]]["scores"]["ndcg_cut_10"]
        if q["pos_token_length"] >= 1024 and 0.4 <= relative_pos <= 0.6:
            alist.append({
                "query": q["text"],
                "doc": corpus_map[qrels_map[q["_id"]]]["text"],
                "rp": relative_pos,
                "token_length": q["pos_token_length"],
                "ndcg_10": ndcg
            })
    blist.extend(alist)

print(len(blist))
model, tokenizer = load_model(model_name)
task = 'Given a web search query, retrieve relevant passages that answer the query'
saliencys = []
for a in tqdm.tqdm(blist):
    query = get_detailed_instruct(task, a["query"])
    long_document = a["doc"]
    tokens, saliency, score = get_saliency_map(model, tokenizer, query, long_document, pooling_method=pooling_type)

    robust_max = np.percentile(saliency, 99)
    if robust_max > 0:
        saliency = np.clip(saliency, 0, robust_max)
        saliency = saliency / robust_max

    saliency = resample_to_fixed_length(saliency)
    saliencys.append(saliency)

with open(f"gradient_saliency/{model_name.split('/')[-1]}.pkl", "wb") as f:
    pickle.dump(saliencys, f)