"""
Write the embeddings from the dataset to torch files that can be loaded quicker

modal run torched.py
"""

from modal import App, Image, Volume, Secret

DATASET_DIR="/embeddings"
VOLUME = "embeddings"
# DIRECTORY = f"{DATASET_DIR}/fineweb-edu-sample-10BT-chunked-500-HF4" 
DIRECTORY = f"{DATASET_DIR}/fineweb-edu-sample-100BT-chunked-500" 
# SAVE_DIRECTORY = f"{DATASET_DIR}/fineweb-edu-sample-10BT-chunked-500-HF4-torched"
SAVE_DIRECTORY = f"{DATASET_DIR}/fineweb-edu-sample-100BT-chunked-500-torched"

# We define our Modal Resources that we'll need
volume = Volume.from_name(VOLUME, create_if_missing=True)
image = Image.debian_slim(python_version="3.9").pip_install(
    "datasets==2.16.1", "apache_beam==2.53.0", "tqdm", "torch", "numpy"
)
app = App(image=image)  # Note: prior to April 2024, "app" was called "stub"

# NUM_EMBEDDINGS = 25504378
# SHARD_SIZE = 262144 # 2048*128

@app.function(
    volumes={DATASET_DIR: volume}, 
    timeout=60000,
    # ephemeral_disk=2145728, # in MiB
)
def torch_dataset_shard(file):
    # Redownload the dataset
    import time
    # from datasets import load_from_disk
    import pandas as pd
    from tqdm import tqdm
    import torch
    import numpy as np
    import os

    print("loading", file)
    # dataset = load_from_disk(DIRECTORY)
    df = pd.read_parquet(f"{DIRECTORY}/train/{file}")
    print("loaded", file)
    # train_dataset = dataset["train"]

    # start_idx = shard * SHARD_SIZE
    # end_idx = min(start_idx + SHARD_SIZE, NUM_EMBEDDINGS)
    # print("reading", shard)
    embeddings = df["embedding"].to_numpy()
    embeddings = np.array([np.array(e).astype(np.float32) for e in embeddings])
    # shard_embeddings = np.array(train_dataset.select(range(start_idx, end_idx))["embedding"])
    # print("permuting", shard)
    # shard_embeddings = np.random.permutation(shard_embeddings)  # {{ edit_1 }}
    shard = file.split(".")[0]
    print("saving", shard)
    shard_tensor = torch.tensor(embeddings, dtype=torch.float32)
    if not os.path.exists(f"{SAVE_DIRECTORY}"):
        os.makedirs(f"{SAVE_DIRECTORY}")
    torch.save(shard_tensor, f"{SAVE_DIRECTORY}/{shard}.pt")
    print("done!", shard)
    volume.commit()
    return shard

@app.local_entrypoint()
def main():
    # num_shards = NUM_EMBEDDINGS // SHARD_SIZE + (1 if NUM_EMBEDDINGS % SHARD_SIZE != 0 else 0)
    # shards = list(range(num_shards))
    # # torch_dataset.remote()
    # for resp in torch_dataset_shard.map(shards, order_outputs=False, return_exceptions=True):
    #     if isinstance(resp, Exception):
    #         print(f"Exception: {resp}")
    #         continue
    #     print(resp)

    files = [f"data-{i:05d}-of-00989.parquet" for i in range(989)]
    files = files[2:]
    # files = [f"data-{i:05d}-of-00099.parquet" for i in range(99)]
    
    # process_dataset.remote(file, max_tokens=MAX_TOKENS, num_cpu=NUM_CPU)
    for resp in torch_dataset_shard.map(files, order_outputs=False, return_exceptions=True):
        if isinstance(resp, Exception):
            print(f"Exception: {resp}")
            continue
        print(resp)


