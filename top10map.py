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

NUM_CPU=8
N=5

DATASET_DIR="/embeddings"
VOLUME = "embeddings"

D_IN = 768 # the dimensions from the embedding models
EXPANSION = 32
SAE = f"64_{EXPANSION}"
DIRECTORY = f"{DATASET_DIR}/fineweb-edu-sample-10BT-chunked-500-HF4-{SAE}" 
SAVE_DIRECTORY = f"{DATASET_DIR}/fineweb-edu-sample-10BT-chunked-500-HF4-{SAE}-top10"

# We define our Modal Resources that we'll need
volume = Volume.from_name(VOLUME, create_if_missing=True)
image = Image.debian_slim(python_version="3.9").pip_install(
    "datasets==2.16.1", "apache_beam==2.53.0", "transformers", "pandas", "tqdm"
)
app = App(image=image)  # Note: prior to April 2024, "app" was called "stub"

def get_top_n_rows_by_top_act(df, top_indices, top_acts, index):
    index_positions = np.where(np.any(top_indices == index, axis=1),
                           np.argmax(top_indices == index, axis=1),
                           -1)
    
    act_values = np.where(index_positions != -1, 
                      top_acts[np.arange(len(top_acts)), index_positions], 
                      0)

    top_5_indices = np.argsort(act_values)[-5:][::-1]
    filtered_df = df.loc[top_5_indices].copy()
    filtered_df['feature'] = index
    filtered_df['activation'] = act_values[top_5_indices]
    return filtered_df

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
    for index in tqdm(feature_chunk, desc=f"Chunk {chunk_index}", position=chunk_index):
        top = get_top_n_rows_by_top_act(df, top_indices, top_acts, index)
        results.append(top)
    return pd.concat(results, ignore_index=True)


@app.function(cpu=NUM_CPU, volumes={DATASET_DIR: volume}, timeout=3000)
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
    # files = [f"data-{i:05d}-of-00989.arrow" for i in range(989)]
    files = [f"data-{i:05d}-of-00099.parquet" for i in range(99)]
    files = files[2:]
    
    # process_dataset.remote(file, max_tokens=MAX_TOKENS, num_cpu=NUM_CPU)
    for resp in process_dataset.map(files, order_outputs=False, return_exceptions=True):
        if isinstance(resp, Exception):
            print(f"Exception: {resp}")
            continue
        print(resp)


