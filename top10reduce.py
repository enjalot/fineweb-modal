from modal import App, Image, Volume, Secret

DATASET_DIR="/embeddings"
VOLUME = "embeddings"
SAE = "64_32"

SAMPLE_DIRECTORY = f"{DATASET_DIR}/fineweb-edu-sample-10BT-chunked-500-HF4-{SAE}-3/train"
DIRECTORY = f"{DATASET_DIR}/fineweb-edu-sample-10BT-chunked-500-HF4-{SAE}-3-top10"
SAVE_DIRECTORY = f"{DATASET_DIR}/fineweb-edu-sample-10BT-chunked-500-HF4-{SAE}-3-top10/combined"




# We define our Modal Resources that we'll need
volume = Volume.from_name(VOLUME, create_if_missing=True)
image = Image.debian_slim(python_version="3.9").pip_install(
    "pandas", "datasets==2.16.1", "apache_beam==2.53.0"
)
app = App(image=image) 

@app.function(
    volumes={DATASET_DIR: volume}, 
    timeout=60000,
    # ephemeral_disk=2145728, # in MiB
)
def populate_indices(samples):
    import pandas as pd

    shard = samples.iloc[0]['shard']
    indices = samples['index'].tolist()

    print("reading shard", shard, len(indices))
    sample_df = pd.read_parquet(f"{SAMPLE_DIRECTORY}/{shard}")
    sample_df = sample_df.iloc[indices].copy()
    sample_df['feature'] = samples['feature'].tolist()
    sample_df['activation'] = samples['activation'].tolist()
    print("returning samples for", shard)

    return sample_df

@app.function(
    volumes={DATASET_DIR: volume}, 
    timeout=60000,
    # ephemeral_disk=2145728, # in MiB
)
def reduce_top10_indices(directory, save_directory, N):
    import os
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    files = [f for f in os.listdir(directory) if f.endswith('.parquet')]
    print("len files", len(files))

    import pandas as pd

    combined_indices_path = f"{save_directory}/combined_indices.parquet"
    if not os.path.exists(combined_indices_path):
        print("creating combined_indices")
        all_dataframes = []
        for file in files:
            print(f"Reading {file}")
            df = pd.read_parquet(f"{directory}/{file}")
            all_dataframes.append(df)

        # Concatenate all DataFrames into a single DataFrame
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        print("combined")
        combined_df.to_parquet(combined_indices_path)
    else:
        print(f"{combined_indices_path} already exists. Loading it.")
        combined_df = pd.read_parquet(combined_indices_path)

    combined_df = combined_df.sort_values(by=['feature', 'activation'], ascending=[True, False])
    combined_df = combined_df.groupby('feature').head(N).reset_index(drop=True)
    print(f"writing top{N}")
    combined_df.to_parquet(f"{save_directory}/combined_indices_top{N}.parquet")
    volume.commit()

    shard_counts = combined_df.groupby('shard').size().reset_index(name='count')
    print("shard_counts", shard_counts.head())

    print("Number of shards:", len(shard_counts))
    rows_by_shard = [combined_df[combined_df['shard'] == shard] for shard in combined_df['shard'].unique()]

    results = []
    for resp in populate_indices.map(rows_by_shard, order_outputs=False, return_exceptions=True):
        if isinstance(resp, Exception):
            print(f"Exception: {resp}")
            continue
        results.append(resp)

    print("concatenating final results")
    final_df = pd.concat(results, ignore_index=True)
    final_df = final_df.drop(columns=['index', '__index_level_0__'], errors='ignore')
    print("sorting final results")
    final_df = final_df.sort_values(by=['feature', 'activation'], ascending=[True, False])
    print("writing final results")
    final_df.to_parquet(f"{save_directory}/samples_top{N}.parquet")
    volume.commit()
    return "done"


    # for resp in reduce_top10.map(pairs, order_outputs=False, return_exceptions=True):
    #     if isinstance(resp, Exception):
    #         print(f"Exception: {resp}")
    #         continue
    #     print(resp)



@app.local_entrypoint()
def main():
    reduce_top10_indices.remote(DIRECTORY, SAVE_DIRECTORY, 10)
    

