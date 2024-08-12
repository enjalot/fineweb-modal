"""
For each of the parquet files with activations, find the top 10 and write to an intermediate file
modal run top10map.py
"""
from modal import App, Image, Volume
import os
import numpy as np

NUM_CPU=1
MAX_TOKENS = 500
OVERLAP = 0.1 # 10% overlap when chunking


DATASET_DIR="/embeddings"
VOLUME = "embeddings"

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

def get_top_n_rows_by_top_act(df, index, n=5):
    # Filter rows where the index is present in the 'top_indices'
    filtered_df = df[df['top_indices'].apply(lambda indices: index in indices)]
    # Extract the corresponding 'top_act' for the given index
    def extract_act(row):
        if index not in row['top_indices']:
            return -1
        index_pos = np.where(row['top_indices'] == index)[0][0]
        return row['top_acts'][index_pos]
    # Apply the function to get the corresponding 'top_act'
    col = filtered_df.apply(extract_act, axis=1)
    filtered_df[f"act_for_{index}"] = col
    filtered_df = filtered_df[filtered_df[f"act_for_{index}"] >= 0]
    # Sort by 'act_for_index' and get the top N rows
    result_df = filtered_df.sort_values(by=f"act_for_{index}", ascending=False).head(n)
    return result_df



@app.function(cpu=NUM_CPU, volumes={DATASET_DIR: volume}, timeout=3000)
def process_dataset(file):
    import time
    from tqdm import tqdm
    import pandas as pd

    start = time.perf_counter()
    # Load the dataset as a Hugging Face dataset
    print(f"Loading dataset from {DIRECTORY}/train/{file}")
    df = pd.read_parquet(f"{DIRECTORY}/train/{file}")
    print("dataset", len(df))
    print(f"Dataset loaded in {time.perf_counter()-start:.2f} seconds for {file}") 

    num_features = 768 * EXPANSION
    top_df = pd.DataFrame()
    for feature in tqdm(range(num_features), desc="Processing features"):
        top_rows = get_top_n_rows_by_top_act(df, index=feature, n=10)
        top_df = top_df.append(top_rows, ignore_index=True)

    if not os.path.exists(SAVE_DIRECTORY):
        os.makedirs(SAVE_DIRECTORY)
    print(f"saving to {SAVE_DIRECTORY}/{file}")
    top_df.to_parquet(f"{SAVE_DIRECTORY}/{file}")
    volume.commit()
    return f"All done with {file}", len(top_df)


@app.local_entrypoint()
def main():
    # files = [f"data-{i:05d}-of-00989.arrow" for i in range(989)]
    files = [f"data-{i:05d}-of-00099.arrow" for i in range(99)]
    
    # process_dataset.remote(file, max_tokens=MAX_TOKENS, num_cpu=NUM_CPU)
    for resp in process_dataset.map(files, order_outputs=False, return_exceptions=True):
        if isinstance(resp, Exception):
            print(f"Exception: {resp}")
            continue
        print(resp)


