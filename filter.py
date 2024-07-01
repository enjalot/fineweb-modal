from modal import App, Image, Volume, Secret

DATASET_DIR="/embeddings"
VOLUME = "embeddings"
DIRECTORY = f"{DATASET_DIR}/fineweb-edu-sample-10BT-chunked-500-HF2" # converted the original to a dataset
SAVE_DIRECTORY = f"{DATASET_DIR}/fineweb-edu-sample-10BT-chunked-500-HF4"

# We define our Modal Resources that we'll need
volume = Volume.from_name(VOLUME, create_if_missing=True)
image = Image.debian_slim(python_version="3.9").pip_install(
    "datasets==2.16.1", "apache_beam==2.53.0"
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
def filter_dataset():
    # Redownload the dataset
    import time
    from datasets import load_from_disk
    print("loading")
    dataset = load_from_disk(DIRECTORY)
    print("filtering")
    filtered = dataset.filter(lambda x: x > 50, input_columns=["chunk_token_count"])
    # print("sorting")
    # dataset.sort(column_names=["id", "chunk_index"], keep_in_memory=True)
    print("saving")
    filtered.save_to_disk(SAVE_DIRECTORY, num_shards={"train":99})
    print("done!")
    volume.commit()


@app.local_entrypoint()
def main():
    filter_dataset.remote()

