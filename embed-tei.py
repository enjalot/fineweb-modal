"""
Embed a dataset using the HuggingFace TEI
"""
import os
import json
import time
import asyncio
import subprocess

from modal import App, Image, Secret, Volume, build, enter, exit, gpu, method

DATASET_DIR = "/data"
# DATASET_SAVE_CHUNKED = f"fineweb-edu-sample-10BT-chunked-500"
DATASET_SAVE_CHUNKED = f"fineweb-edu-sample-100BT-chunked-500"
EMBEDDING_DIR = "/embeddings"

# We first set out configuration variables for our script.
## Embedding Containers Configuration
# GPU_CONCURRENCY = 100
MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"
MODEL_SLUG = MODEL_ID.split("/")[-1]

MODEL_DIR = "/model"
MODEL_REVISION="main"

GPU_CONCURRENCY = 10
GPU_CONFIG = gpu.A10G()
GPU_IMAGE = "ghcr.io/huggingface/text-embeddings-inference:86-1.2"
# GPU_CONFIG = gpu.A100(size="40GB")
# GPU_CONFIG = gpu.A100(size="80GB")
# GPU_IMAGE = "ghcr.io/huggingface/text-embeddings-inference:1.2"
# GPU_CONFIG = gpu.H100()
# GPU_IMAGE = "ghcr.io/huggingface/text-embeddings-inference:hopper-1.2"


SENTENCE_TOKEN_LIMIT = 512
CLIENT_BATCH_TOKEN_LIMIT = 768 * SENTENCE_TOKEN_LIMIT #A10G
SERVER_BATCH_TOKEN_LIMIT = 2*768 * SENTENCE_TOKEN_LIMIT #A10G
# CLIENT_BATCH_TOKEN_LIMIT =  512 * SENTENCE_TOKEN_LIMIT #A100 40GB
# SERVER_BATCH_TOKEN_LIMIT = 4*2048 * SENTENCE_TOKEN_LIMIT #A100 40GB

LAUNCH_FLAGS = [
    "--model-id",
    MODEL_ID,
    "--port",
    "8000",
    "--max-client-batch-size",
    str(10000), # lots of small sentences can have large batch
    "--max-batch-tokens",
    str(SERVER_BATCH_TOKEN_LIMIT),
    "--auto-truncate",
    "--dtype",
    "float16"
]

## Dataset-Specific Configuration
DATASET_READ_VOLUME = Volume.from_name(
    "embedding-fineweb-edu", create_if_missing=True
)
EMBEDDING_CHECKPOINT_VOLUME = Volume.from_name(
    "embeddings", create_if_missing=True
)

def spawn_server() -> subprocess.Popen:
    import socket

    process = subprocess.Popen(["text-embeddings-router"] + LAUNCH_FLAGS)
    # Poll until webserver at 127.0.0.1:8000 accepts connections before running inputs.
    while True:
        try:
            socket.create_connection(("127.0.0.1", 8000), timeout=1).close()
            print("Webserver ready!")
            return process
        except (socket.timeout, ConnectionRefusedError):
            # Check if launcher webserving process has exited.
            # If so, a connection can never be made.
            retcode = process.poll()
            if retcode is not None:
                raise RuntimeError(
                    f"launcher exited unexpectedly with code {retcode}"
                )


tei_image = (
    Image.from_registry(
        GPU_IMAGE,
        add_python="3.10",
    )
    .dockerfile_commands("ENTRYPOINT []")
    .pip_install("httpx", "numpy")
)

with tei_image.imports():
    import numpy as np

app = App(
    "fineweb-embeddings-tei"
)  

@app.cls(
    gpu=GPU_CONFIG,
    image=tei_image,
    concurrency_limit=GPU_CONCURRENCY,
    allow_concurrent_inputs=True,
    retries=3,
)
class TextEmbeddingsInference:
    @build()
    def download_model(self):
        spawn_server()

    @enter()
    def open_connection(self):
        # If the process is running for a long time, the client does not seem to close the connections, results in a pool timeout
        from httpx import AsyncClient

        self.process = spawn_server()
        self.client = AsyncClient(base_url="http://127.0.0.1:8000", timeout=30)

    @exit()
    def terminate_connection(self):
        self.process.terminate()

    @method()
    # async def _embed(self, chunk_batch):
    async def embed(self, chunk_batch):
        texts = chunk_batch[0]
        # print("TEXTS", len(texts))
        res = await self.client.post("/embed", json={"inputs": texts})
        # try:
        emb = res.json()
        # print("SUP EMB", len(emb))
        return chunk_batch, np.array(emb)
        #   print("res", len(emb))
        # except Exception as e:
        #   print("RES", res)
        #   print("ERROR", e)
        #   print("IDS", chunk_batch[1])

    # @method()
    # async def embed(self, batch):
    # #     # """Embeds a list of texts.  id, text = chunks[1]"""
    #     coros = [
    #         self._embed(batch)
    #     ]
    #     embeddings = np.vstack(await asyncio.gather(*coros))
    #     return batch, embeddings
  

@app.function(
    concurrency_limit=10,
    image=Image.debian_slim().pip_install(
        "pandas", "pyarrow", "tqdm"
    ),
    volumes={
        DATASET_DIR: DATASET_READ_VOLUME,
        EMBEDDING_DIR: EMBEDDING_CHECKPOINT_VOLUME,
    },
    timeout=86400,
    secrets=[Secret.from_name("huggingface-secret")],
)
def batch_loader(file):
    import pandas as pd
    from tqdm import tqdm
    import time


    print(f"reading in {file}")
    file_path = f"{DATASET_DIR}/{DATASET_SAVE_CHUNKED}/train/{file}"
    df = pd.read_parquet(file_path)
    print(f"sorting {file}", len(df))
    df = df.sort_values(by='chunk_token_count', ascending=True)
    # Filter df to only rows with chunk_token_count > 50
    # TODO: this shouldn't be needed if chunker is run with fix (but 10BT and 100BT were chunked without it)
    df = df[df['chunk_token_count'] > 50]
    print(f"Filtered {file} to {len(df)} rows with chunk_token_count > 50")
    # batches = []
    batches_text = []
    current_batch = []
    current_batch_counts = []
    current_batch_text = []
    # current_token_count = 0
    batch_indices = []
    current_batch_indices = []
    # attention_masks = []  # List to store attention masks for each batch
    packed = []


    # Tokenized version of "clustering: "
    prefix = [101, 9324, 2075, 1024]

    print("building batches for ", file, "with client batch token limit", CLIENT_BATCH_TOKEN_LIMIT)
    start = time.monotonic_ns()
    
    for index, row in df.iterrows():
        chunk_token_count = row['chunk_token_count'] + 4 # 4 for the prefix
        # chunk = prefix + list(row['chunk_tokens'])
        chunkt = "clustering: " + row['chunk_text']
        # proposed_batch = current_batch + [chunk]
        proposed_batch_count = current_batch_counts + [chunk_token_count]
        # proposed_length = max(len(tokens) for tokens in proposed_batch) * len(proposed_batch)
        proposed_length = max(count for count in proposed_batch_count) * len(proposed_batch_count)

        if proposed_length <= CLIENT_BATCH_TOKEN_LIMIT:
            # current_batch.append(chunk)
            current_batch_text.append(chunkt)
            current_batch_indices.append(index)
            current_batch_counts.append(chunk_token_count)
            # current_token_count = proposed_length
        else:
            # Pad the current batch
            # max_length = max(len(tokens) for tokens in current_batch)
            # padded_batch = [tokens + [0] * (max_length - len(tokens)) for tokens in current_batch]
            # attention_mask = [[1] * len(tokens) + [0] * (max_length - len(tokens)) for tokens in current_batch]
            # batches.append(padded_batch)
            batches_text.append(current_batch_text)
            # attention_masks.append(attention_mask)
            batch_indices.append(current_batch_indices)
            # Start new batch
            # current_batch = [chunk]
            current_batch_counts = [chunk_token_count]
            current_batch_text = [chunkt]
            current_batch_indices = [index]
            # current_token_count = len(chunk)

    if current_batch_counts:
        # Pad the final batch
        # max_length = max(len(tokens) for tokens in current_batch)
        # padded_batch = [tokens + [0] * (max_length - len(tokens)) for tokens in current_batch]
        # attention_mask = [[1] * len(tokens) + [0] * (max_length - len(tokens)) for tokens in current_batch]

        # batches.append(padded_batch)
        batch_indices.append(current_batch_indices)
        batches_text.append(current_batch_text)


    # print("length of first batch", len(batches[0]))
    # first_batch_length = sum(len(chunk) for chunk in batches[0])
    # print("Total length of all elements in the first batch:", first_batch_length)
    # print(f"number of batches {len(batches)}")

    duration_s = (time.monotonic_ns() - start) / 1e9
    print(f"batched {file} in {duration_s:.0f}s")

    # print("BATCHES TEXT", batches_text[0:3])
    # print("BATCHES INDICES", batch_indices[0:3])
    responses = []
    for batch_text, batch_indices in zip(batches_text, batch_indices):
        packed.append((batch_text, batch_indices))

    print(f"{len(packed)} batches")

    pbar = tqdm(total=len(packed), desc=f"embedding {file}")
    model = TextEmbeddingsInference()

    for resp in model.embed.map(
        packed,
        order_outputs=False, 
        return_exceptions=False
    ):
        responses.append(resp)
        pbar.update(1)

    # Ensure the 'embedding' column exists
    if 'embedding' not in df.columns:
        df['embedding'] = None  # Initialize the column with None or an appropriate default value

    print("zipping batches with responses")
    for batch, response in responses:
        # print("RESPONSE!", response)
        # print("RESPONSE", len(response))
        for idx, embedding in zip(batch[1], response):
            df.at[idx, 'embedding'] = embedding
    
    print(f"writing out embeddings file {file}")
    if not os.path.exists(f"{EMBEDDING_DIR}/{DATASET_SAVE_CHUNKED}/train"):
        os.makedirs(f"{EMBEDDING_DIR}/{DATASET_SAVE_CHUNKED}/train", exist_ok=True)
    df.to_parquet(f"{EMBEDDING_DIR}/{DATASET_SAVE_CHUNKED}/train/{file}")
    EMBEDDING_CHECKPOINT_VOLUME.commit()
    return f"done with {file}"

@app.local_entrypoint()
def full_job():

    # file = "data-00000-of-00099.parquet"
    # file = "data-00004-of-00099.parquet"

    files = [f"data-{i:05d}-of-00989.parquet" for i in range(522,990)]
    # files = ["data-00097-of-00989.parquet"]
    # for file in files:
        # print("kicking off", file)
        # batch_loader.remote(file=file, batch_size = BATCH_TOKEN_LIMIT)
    for resp in batch_loader.map(
        files,
        order_outputs=False, 
        return_exceptions=True
    ):
        print(resp)

    # batch_loader.remote(file=file, batch_size = BATCH_TOKEN_LIMIT)
    print("done")

