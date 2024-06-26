from modal import App, Image, Volume

# We first set out configuration variables for our script.
DATASET_DIR = "/data"
DATASET_NAME = "HuggingFaceFW/fineweb-edu"
SAMPLE = "100BT"
# DATASET_FILES = "sample/10BT/*.parquet"
DATASET_FILES = f"sample/{SAMPLE}/*.parquet"
DATASET_SAVE =f"fineweb-edu-sample-{SAMPLE}"


# We define our Modal Resources that we'll need
volume = Volume.from_name("embedding-fineweb-edu", create_if_missing=True)
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
    ephemeral_disk=2145728 # in MiB
)
def download_dataset():
    # Redownload the dataset
    import time

    from datasets import load_dataset

    start = time.time()
    dataset = load_dataset(DATASET_NAME,  data_files=DATASET_FILES, num_proc=6)
    end = time.time()
    print(f"Download complete - downloaded files in {end-start}s")

    dataset.save_to_disk(f"{DATASET_DIR}/{DATASET_SAVE}")
    volume.commit()

@app.function(volumes={DATASET_DIR: volume})
def load_dataset():
    import time

    from datasets import load_from_disk

    start = time.perf_counter()
    # Load the dataset as a Hugging Face dataset
    print(f"Loading dataset from {DATASET_DIR}/{DATASET_SAVE}")
    dataset = load_from_disk(f"{DATASET_DIR}/{DATASET_SAVE}")
    print(f"Dataset loaded in {time.perf_counter()-start:.2f} seconds") 


    # # Sample the dataset to 100,000 rows
    # print("Sampling dataset to 100,000 rows")
    # sampled_datasets = dataset["train"].select(range(100000))
    # sampled_datasets.save_to_disk(f"{DATASET_DIR}/{DATASET_SAVE}-100k")


# TODO: make a function to delete files
# the 00099 files are old/wrong

# TODO: make a function to load a single file from dataset

@app.local_entrypoint()
def main():
    download_dataset.remote()
    # load_dataset.remote()

