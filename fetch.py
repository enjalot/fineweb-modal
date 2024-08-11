"""
fetch a file from a modal volume and write it locally
"""

from modal import App, Image, Volume

# We first set out configuration variables for our script.
DATASET_DIR = "/data"
# DATASET_DIR = "/embeddings"
# DATASET_NAME = "HuggingFaceFW/fineweb-edu"
# DATASET_FILES = "sample/10BT/*.parquet"
DATASET_SAVE ="fineweb-edu-sample-10BT"
MAX_TOKENS = 500
# DATASET_SAVE = f"fineweb-edu-sample-10BT-chunked-{MAX_TOKENS}"
DATASET_SAVE = f"fineweb-edu-sample-10BT"
DIRECTORY = f"{DATASET_DIR}/{DATASET_SAVE}/train"

# MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"

# We define our Modal Resources that we'll need
volume = Volume.from_name("embedding-fineweb-edu", create_if_missing=True)
# volume = Volume.from_name("embeddings", create_if_missing=True)
image = Image.debian_slim(python_version="3.9").pip_install(
    "datasets==2.16.1", "apache_beam==2.53.0", "transformers", "pandas", "tqdm"
)
app = App(image=image)  # Note: prior to April 2024, "app" was called "stub"


@app.function(volumes={DATASET_DIR: volume}, timeout=3000)
def fetch_dataset(file):
    import pandas as pd
    from datasets import load_dataset
    print("loading", file)
    # Load the dataset as a Hugging Face dataset
    # df = pd.read_parquet(file)
    dataset = load_dataset("arrow", data_files=file)
    df = pd.DataFrame(dataset['train'])
    print("file loaded, returning", file)
    return df

@app.local_entrypoint()
def main():
    import pandas as pd

    file = "data-00000-of-00099.arrow"
    # file = "data-00000-of-00099.parquet"
    file_path = f"{DIRECTORY}/{file}"
    resp = fetch_dataset.remote(file_path)
    if isinstance(resp, Exception):
        print(f"Exception: {resp}")
    else:
        print(resp)
        resp.to_parquet(f"./notebooks/{file}")
        