from modal import App, Image, Volume, Secret

DATASET_DIR="/embeddings"
VOLUME = "embeddings"
DIRECTORY = f"{DATASET_DIR}/fineweb-edu-sample-10BT-chunked-500-HF4-64_32-top10"

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
def reduce_top10(files, N):
    import pandas as pd

    df = pd.read_parquet(f"{DIRECTORY}/{files[0]}")
    for file in files[1:]:
        print("loading", file)
        temp_df = pd.read_parquet(f"{DIRECTORY}/{file}")
        print("concat")
        df = pd.concat([df, temp_df], ignore_index=True)
        print("grouping")
        df = df.groupby('feature').apply(lambda x: x.nlargest(N, 'activation')).reset_index(drop=True)
        df.to_parquet(f"{DIRECTORY}/top{N}.parquet")
    volume.commit()
    return "done" 




@app.local_entrypoint()
def main():
    files = [f"data-{i:05d}-of-00099.parquet" for i in range(99)]
    reduce_top10.remote(files, 10)
    

