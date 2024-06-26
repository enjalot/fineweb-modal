import os
import json
import time
import asyncio
import subprocess

from modal import App, Image, Secret, Volume, build, enter, exit, gpu, method

DATASET_DIR = "/data"
DATASET_SAVE_CHUNKED = f"fineweb-edu-sample-10BT-chunked-500"
CHECKPOINT_DIR = "/checkpoint"

# We first set out configuration variables for our script.
## Embedding Containers Configuration
# GPU_CONCURRENCY = 100
MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"
MODEL_SLUG = MODEL_ID.split("/")[-1]

MODEL_DIR = "/model"
MODEL_REVISION="main"

GPU_CONCURRENCY = 10
# GPU_CONFIG = gpu.A100(size="80GB")
# GPU_CONFIG = gpu.A100(size="40GB")
# GPU_CONFIG = gpu.A10G()
GPU_CONFIG = gpu.H100()


## Dataset-Specific Configuration
DATASET_READ_VOLUME = Volume.from_name(
    "embedding-fineweb-edu", create_if_missing=True
)
EMBEDDING_CHECKPOINT_VOLUME = Volume.from_name(
    "embeddings", create_if_missing=True
)
def download_model_to_image(model_dir, model_name, model_revision):
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(model_dir, exist_ok=True)

    snapshot_download(
        repo_id=model_name,
        revision=model_revision,
        local_dir=model_dir,
        ignore_patterns=["*.pt", "*.bin"],  # Using safetensors
    )
    move_cache()

st_image = (
    Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.1.2",
        "numpy==1.26.3",
        "transformers==4.39.3",
        "hf-transfer==0.1.6",
        "huggingface_hub==0.22.2",
        "einops==0.7.0"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        download_model_to_image,
        timeout=60 * 20,
        kwargs={
            "model_dir": MODEL_DIR,
            "model_name": MODEL_ID,
            "model_revision": MODEL_REVISION,
        },
        secrets=[Secret.from_name("huggingface-secret")],
    )
)
with st_image.imports():
    import numpy as np
    import torch
    from torch.cuda.amp import autocast
    from transformers import AutoTokenizer, AutoModel

app = App(
    "fineweb-embeddings-st"
)  

@app.cls(
    gpu=GPU_CONFIG,
    # cpu=16,
    concurrency_limit=GPU_CONCURRENCY,
    timeout=60 * 10,
    container_idle_timeout=60 * 10,
    allow_concurrent_inputs=1,
    image=st_image,
)
class TransformerModel:
    @enter()
    def start_engine(self):
        # import torch
        # from transformers import AutoTokenizer, AutoModel

        self.device = torch.device("cuda")

        print("🥶 cold starting inference")
        start = time.monotonic_ns()

        self.model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, safe_serialization=True)#, rotary_scaling_factor=2 )
        # self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", model_max_length=512) # MAX_TOKENS
        self.model.to(self.device)
        self.model.eval()

        # print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1e6} MB")
        duration_s = (time.monotonic_ns() - start) / 1e9
        print(f"🏎️ engine started in {duration_s:.0f}s")

    @method()
    def embed(self, batch_mask_index):
        batch, mask, index = batch_mask_index
        # print(torch.cuda.memory_summary(device=self.device, abbreviated=True))

        tokens_tensor = torch.tensor(batch)
        attention_mask = torch.tensor(mask)

        encoded_input = {
            'input_ids': tokens_tensor.to(self.device),
            'attention_mask': attention_mask.to(self.device)
        }
        # encoded_input = {key: value.to(self.device) for key, value in inputs}
        start = time.monotonic_ns()
        with torch.no_grad():#, autocast():
            model_output = self.model(**encoded_input)
            embeddings = model_output[0][:, 0]
            normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            normalized_embeddings_cpu = normalized_embeddings.cpu().numpy()

            duration_ms = (time.monotonic_ns() - start) / 1e6
            print(f"embedding took {duration_ms:.0f}ms")
            
            del encoded_input
            del model_output
            del embeddings
            del normalized_embeddings
            torch.cuda.empty_cache()

            # print(torch.cuda.memory_summary(device=self.device, abbreviated=True))
            return index, normalized_embeddings_cpu



@app.function(
    image=Image.debian_slim().pip_install(
        "pandas", "pyarrow", "tqdm"
    ),
    volumes={
        DATASET_DIR: DATASET_READ_VOLUME,
        CHECKPOINT_DIR: EMBEDDING_CHECKPOINT_VOLUME,
    },
    timeout=86400,
    secrets=[Secret.from_name("huggingface-secret")],
)
def batch_loader(file, batch_size: int = 512 * 1024):
    import pandas as pd
    from tqdm import tqdm
    import time


    print(f"reading in {file}")
    file_path = f"{DATASET_DIR}/{DATASET_SAVE_CHUNKED}/train/{file}"
    df = pd.read_parquet(file_path)
    print(f"sorting {file}")
    df = df.sort_values(by='chunk_token_count', ascending=True)
    batches = []
    current_batch = []
    current_token_count = 0
    batch_indices = []
    current_batch_indices = []
    attention_masks = []  # List to store attention masks for each batch


    # Tokenized version of "clustering: "
    prefix = [101, 9324, 2075, 1024]

    print("building batches for ", file)
    start = time.monotonic_ns()
    
    for index, row in df.iterrows():
        # chunk_token_count = row['chunk_token_count']
        chunk = prefix + list(row['chunk_tokens'])
        proposed_batch = current_batch + [chunk]
        proposed_length = max(len(tokens) for tokens in proposed_batch) * len(proposed_batch)

        if proposed_length <= batch_size:
            current_batch.append(chunk)
            current_batch_indices.append(index)
            # current_token_count = proposed_length
        else:
            # Pad the current batch
            max_length = max(len(tokens) for tokens in current_batch)
            padded_batch = [tokens + [0] * (max_length - len(tokens)) for tokens in current_batch]
            attention_mask = [[1] * len(tokens) + [0] * (max_length - len(tokens)) for tokens in current_batch]
            batches.append(padded_batch)
            attention_masks.append(attention_mask)
            batch_indices.append(current_batch_indices)
            # Start new batch
            current_batch = [chunk]
            current_batch_indices = [index]
            # current_token_count = len(chunk)

    if current_batch:
        # Pad the final batch
        max_length = max(len(tokens) for tokens in current_batch)
        padded_batch = [tokens + [0] * (max_length - len(tokens)) for tokens in current_batch]
        attention_mask = [[1] * len(tokens) + [0] * (max_length - len(tokens)) for tokens in current_batch]

        batches.append(padded_batch)
        batch_indices.append(current_batch_indices)


    print("length of first batch", len(batches[0]))
    first_batch_length = sum(len(chunk) for chunk in batches[0])
    print("Total length of all elements in the first batch:", first_batch_length)
    print(f"number of batches {len(batches)}")

    duration_s = (time.monotonic_ns() - start) / 1e9
    print(f"batched {file} in {duration_s:.0f}s")

    pbar = tqdm(total=len(batches), desc=f"embedding {file}")
    model = TransformerModel()

    responses = []
    for resp in model.embed.map(
        zip(batches, attention_masks, batch_indices), 
        order_outputs=False, 
        return_exceptions=False
    ):
        responses.append(resp)
        pbar.update(1)

    print("zipping batches with responses")
    for batch_idx, response in responses:
        for idx, embedding in zip(batch_idx, response):
            df.at[idx, 'embedding'] = embedding
    
    if not os.path.exists(f"{CHECKPOINT_DIR}/{DATASET_SAVE_CHUNKED}/train"):
        os.makedirs(f"{CHECKPOINT_DIR}/{DATASET_SAVE_CHUNKED}/train", exist_ok=True)
    df.to_parquet(f"{CHECKPOINT_DIR}/{DATASET_SAVE_CHUNKED}/train/{file}")
    return f"done with {file}"

@app.local_entrypoint()
def full_job():

    file = "data-00000-of-00099.parquet"

    batch_loader.remote(file=file, batch_size = (1024) * 512)
    print("done")

