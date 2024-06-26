from modal import App, Image, Volume, Secret

DATASET_DIR="/embeddings"
VOLUME = "embeddings"
DIRECTORY = f"{DATASET_DIR}/fineweb-edu-sample-10BT-chunked-500"
SAVE_DIRECTORY = f"{DIRECTORY}-HF2"

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
    timeout=6000,
    # ephemeral_disk=2145728, # in MiB
    secrets=[Secret.from_name("huggingface-secret")],
)
def convert_dataset():
    # Redownload the dataset
    import time
    from datasets import load_dataset
    print("loading")
    dataset = load_dataset("parquet", data_files=f"{DIRECTORY}/train/*.parquet")
    print("saving")
    dataset.save_to_disk(SAVE_DIRECTORY, num_shards={"train":99})
    print("done!")
    volume.commit()


@app.local_entrypoint()
def main():
    convert_dataset.remote()

