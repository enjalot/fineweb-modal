"""
Write the embeddings from the dataset to torch files that can be loaded quicker
"""

from modal import App, Image, Volume, Secret

DATASET_DIR="/embeddings"
VOLUME = "embeddings"
DIRECTORY = f"{DATASET_DIR}/fineweb-edu-sample-10BT-chunked-500-HF4" 
SAVE_DIRECTORY = f"{DATASET_DIR}/fineweb-edu-sample-10BT-chunked-500-HF4-torched"

# We define our Modal Resources that we'll need
volume = Volume.from_name(VOLUME, create_if_missing=True)
image = Image.debian_slim(python_version="3.9").pip_install(
    "datasets==2.16.1", "apache_beam==2.53.0", "tqdm", "torch", "numpy"
)
app = App(image=image)  # Note: prior to April 2024, "app" was called "stub"


# The default timeout is 5 minutes re: https://modal.com/docs/guide/timeouts#handling-timeouts
#  but we override this to
# 6000s to avoid any potential timeout issues
@app.function(
    volumes={DATASET_DIR: volume}, 
    timeout=60000,
    # ephemeral_disk=2145728, # in MiB
)
def torch_dataset(shard_size=262144): # 2048*128
    # Redownload the dataset
    import time
    from datasets import load_from_disk
    from tqdm import tqdm
    import torch
    import numpy as np

    print("loading")
    dataset = load_from_disk(DIRECTORY)
    print("loaded")
    train_dataset = dataset["train"]
    num_embeddings = train_dataset.num_rows
    print("num embeddings", num_embeddings)

    # embeddings = train_dataset["embedding"]
    
    num_shards = num_embeddings // shard_size + (1 if num_embeddings % shard_size != 0 else 0)
    print("num_shards", num_shards)

    def get_embeddings_in_chunks():
      for start_idx in range(0, num_embeddings, shard_size):
          end_idx = min(start_idx + shard_size, num_embeddings)
          yield np.array(train_dataset["embedding"][start_idx:end_idx])
        
    for i, shard_embeddings in tqdm(enumerate(get_embeddings_in_chunks()), total=num_shards):
        shard_embeddings = np.random.permutation(shard_embeddings)  # {{ edit_1 }}
        shard_tensor = torch.tensor(shard_embeddings, dtype=torch.float32)
        torch.save(shard_tensor, f"{SAVE_DIRECTORY}/shard_{i:05d}.pt")

    print("done!")
    volume.commit()


NUM_EMBEDDINGS = 25504378
SHARD_SIZE = 262144

@app.function(
    volumes={DATASET_DIR: volume}, 
    timeout=60000,
    # ephemeral_disk=2145728, # in MiB
)
def torch_dataset_shard(shard):
    # Redownload the dataset
    import time
    from datasets import load_from_disk
    from tqdm import tqdm
    import torch
    import numpy as np
    import os

    print("loading", shard)
    dataset = load_from_disk(DIRECTORY)
    print("loaded", shard)
    train_dataset = dataset["train"]

    start_idx = shard * SHARD_SIZE
    end_idx = min(start_idx + SHARD_SIZE, NUM_EMBEDDINGS)
    print("reading", shard)
    shard_embeddings = np.array(train_dataset.select(range(start_idx, end_idx))["embedding"])
    print("permuting", shard)
    shard_embeddings = np.random.permutation(shard_embeddings)  # {{ edit_1 }}
    print("saving", shard)
    shard_tensor = torch.tensor(shard_embeddings, dtype=torch.float32)
    if not os.path.exists(f"{SAVE_DIRECTORY}"):
        os.makedirs(f"{SAVE_DIRECTORY}")
    torch.save(shard_tensor, f"{SAVE_DIRECTORY}/shard_{shard:05d}.pt")
    print("done!", shard)
    volume.commit()
    return shard

@app.local_entrypoint()
def main():
    num_shards = NUM_EMBEDDINGS // SHARD_SIZE + (1 if NUM_EMBEDDINGS % SHARD_SIZE != 0 else 0)
    shards = list(range(num_shards))
    # torch_dataset.remote()
    for resp in torch_dataset_shard.map(shards, order_outputs=False, return_exceptions=True):
        if isinstance(resp, Exception):
            print(f"Exception: {resp}")
            continue
        print(resp)


