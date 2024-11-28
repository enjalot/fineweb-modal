"""
For each of the parquet files with activations, find the top 10 and write to an intermediate file
modal run top10map.py
"""
from modal import App, Image, Volume
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import concurrent.futures
from functools import partial

NUM_CPU=4

N=5 # the number of samples to keep per feature

DATASET_DIR="/embeddings"
VOLUME = "embeddings"

D_IN = 768 # the dimensions from the embedding models
K=64
# EXPANSION = 128
EXPANSION = 32
SAE = f"{K}_{EXPANSION}"
DIRECTORY = f"{DATASET_DIR}/fineweb-edu-sample-10BT-chunked-500-HF4-{SAE}-3" 
SAVE_DIRECTORY = f"{DATASET_DIR}/fineweb-edu-sample-10BT-chunked-500-HF4-{SAE}-3-top10"

files = [f"data-{i:05d}-of-00099.parquet" for i in range(99)]

# We define our Modal Resources that we'll need
volume = Volume.from_name(VOLUME, create_if_missing=True)
image = Image.debian_slim(python_version="3.9").pip_install(
    "datasets==2.16.1", "apache_beam==2.53.0", "transformers", "pandas", "tqdm"
)
app = App(image=image)  # Note: prior to April 2024, "app" was called "stub"

def get_top_n_rows_by_top_act(file, top_indices, top_acts, feature):
    # feature_positions = np.where(np.any(top_indices == feature, axis=1),
    #                        np.argmax(top_indices == feature, axis=1),
    #                        -1)
    # act_values = np.where(feature_positions != -1, 
    #                   top_acts[np.arange(len(top_acts)), feature_positions], 
    #                   0)
    # top_n_indices = np.argsort(act_values)[-N:][::-1]

    # Find positions where feature appears (returns a boolean mask)
    feature_mask = top_indices == feature
    
    # Get the activation values where the feature appears (all others will be 0)
    act_values = np.where(feature_mask.any(axis=1),
                         top_acts[feature_mask].reshape(-1),
                         0)
    
    # Use partition to get top N indices efficiently
    top_n_indices = np.argpartition(act_values, -N)[-N:]
    # Sort just the top N indices
    top_n_indices = top_n_indices[np.argsort(act_values[top_n_indices])[::-1]]

    filtered_df = pd.DataFrame({
        "shard": file,
        "index": top_n_indices,
        "feature": feature,
        "activation": act_values[top_n_indices]
    })
    return filtered_df

# TODO: this uses an incredible amount of memory
# maybe there is a way to reduce memory usage
# for one thing, i'm loading the whole df into memory for each thread (which is 2gb file)
def process_feature_chunk(file, feature_chunk, chunk_index):
    start = time.perf_counter()
    print(f"Loading dataset from {DIRECTORY}/train/{file}", chunk_index)
    df = pd.read_parquet(f"{DIRECTORY}/train/{file}")
    if 'chunk_tokens' in df.columns:
        df.drop(columns=['chunk_tokens'], inplace=True)
    print(f"Dataset loaded in {time.perf_counter()-start:.2f} seconds for {file}", chunk_index) 

    top_indices = np.array(df['top_indices'].tolist())
    top_acts = np.array(df['top_acts'].tolist())

    print(f"top_indices shape: {top_indices.shape}")
    print(f"top_acts shape: {top_acts.shape}")

    print("got numpy arrays", chunk_index)
    
    results = []
    for feature in tqdm(feature_chunk, desc=f"Chunk {chunk_index}", position=chunk_index):
        top = get_top_n_rows_by_top_act(file, top_indices, top_acts, feature)
        results.append(top)
    return pd.concat(results, ignore_index=True)


@app.function(cpu=NUM_CPU, volumes={DATASET_DIR: volume}, timeout=6000)
def process_dataset(file):
    from concurrent.futures import ProcessPoolExecutor, as_completed

    num_features = D_IN * EXPANSION

    # Split the features into chunks for parallel processing
    chunk_size = num_features // NUM_CPU
    feature_chunks = [range(i, min(i + chunk_size, num_features)) for i in range(0, num_features, chunk_size)]

    with ProcessPoolExecutor(max_workers=NUM_CPU) as executor:
        futures = [executor.submit(process_feature_chunk, file, chunk, i) for i, chunk in enumerate(feature_chunks)]
        
        results = []
        i = 0
        for future in as_completed(futures):
            print(f"Processing result {i}")
            i += 1
            results.append(future.result())

    print("concatenating")
    top_df = pd.concat(results, ignore_index=True)
    if not os.path.exists(SAVE_DIRECTORY):
        os.makedirs(SAVE_DIRECTORY)
    print(f"saving to {SAVE_DIRECTORY}/{file}")
    top_df.to_parquet(f"{SAVE_DIRECTORY}/{file}")
    volume.commit()
    return f"All done with {file}", len(top_df)


@app.local_entrypoint()
def main():
    for resp in process_dataset.map(files, order_outputs=False, return_exceptions=True):
        if isinstance(resp, Exception):
            print(f"Exception: {resp}")
            continue
        print(resp)


