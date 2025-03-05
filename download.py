"""
Download a dataset from HuggingFace to a modal volume
s"""
from modal import App, Image, Volume, Secret

# We first set out configuration variables for our script.
VOLUME = "datasets"
DATASET_DIR = "/data"

HF_CACHE_DIR = f"{DATASET_DIR}/cache"

# DATASET_NAME = "HuggingFaceFW/fineweb-edu"
# SAMPLE = "100BT"
# DATASET_FILES = f"sample/{SAMPLE}/*.parquet"
# DATASET_SAVE =f"fineweb-edu-sample-{SAMPLE}"
# VOLUME = "embedding-fineweb-edu"


# DATASET_NAME = "togethercomputer/RedPajama-Data-V2"
# DATASET_SAVE = "RedPajama-Data-V2-sample-10B"
# DATASET_SAMPLE = "sample-10B"
# DATASET_FILES = None

# DATASET_NAME = "monology/pile-uncopyrighted"
# DATASET_SAVE = "pile-uncopyrighted"
# DATASET_SAMPLE = None
# DATASET_FILES = None

# DATASET_NAME = "PleIAs/common_corpus"
# DATASET_SAVE = "common_corpus"
# DATASET_SAMPLE = None
# DATASET_FILES = None

# DATASET_NAME = "bigcode/the-stack-dedup"
# DATASET_SAVE = "the-stack-dedup"
# DATASET_FILES = None

# DATASET_NAME = "wikimedia/wikipedia"
# DATASET_SAVE = "wikipedia-en"
# DATASET_SAMPLE = "20231101.en"
# DATASET_FILES = None

DATASET_NAME = "MedRAG/pubmed"
DATASET_SAVE = "medrag-pubmed"
DATASET_SAMPLE = None
DATASET_FILES = None




# We define our Modal Resources that we'll need
volume = Volume.from_name(VOLUME, create_if_missing=True)
image = Image.debian_slim(python_version="3.9").pip_install(
    "datasets==3.2.0"
)
app = App(image=image)  # Note: prior to April 2024, "app" was called "stub"


# The default timeout is 5 minutes re: https://modal.com/docs/guide/timeouts#handling-timeouts
#  but we override this to
# 6000s to avoid any potential timeout issues
@app.function(
    volumes={DATASET_DIR: volume}, 
    timeout=60000,
    ephemeral_disk=int(3145728), # in MiB
    secrets=[Secret.from_name("huggingface-secret")],
)
def download_dataset():
    # Redownload the dataset
    import time
    import os

    # Set HF cache environment variable
    os.environ['HF_HOME'] = HF_CACHE_DIR
    

    from datasets import load_dataset, DownloadConfig, logging
    logging.set_verbosity_debug()

    start = time.time()
    if DATASET_FILES:
        dataset = load_dataset(DATASET_NAME,  data_files=DATASET_FILES, num_proc=6, trust_remote_code=True, download_config=DownloadConfig(resume_download=True, cache_dir=HF_CACHE_DIR))
    elif DATASET_SAMPLE:
        dataset = load_dataset(DATASET_NAME,  DATASET_SAMPLE, num_proc=6, trust_remote_code=True, download_config=DownloadConfig(resume_download=True, cache_dir=HF_CACHE_DIR))
    else:
        dataset = load_dataset(DATASET_NAME,  num_proc=6, trust_remote_code=True, download_config=DownloadConfig(resume_download=True, cache_dir=HF_CACHE_DIR))
    end = time.time()
    print(f"Download complete - downloaded files in {end-start}s")

    dataset.save_to_disk(f"{DATASET_DIR}/{DATASET_SAVE}")
    volume.commit()

@app.function(volumes={DATASET_DIR: volume})
def load_dataset():
    import time
    import os

    # Set HF cache environment variable
    os.environ['HF_HOME'] = HF_CACHE_DIR
    

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

