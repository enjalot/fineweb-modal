from modal import App, Image, Volume, Secret

# We first set out configuration variables for our script.
DATASET_DIR = "/embeddings"
VOLUME="embeddings"
HF_REPO="enjalot/fineweb-edu-sample-10BT-chunked-500-nomic-text-v1.5-2"
DIRECTORY = f"{DATASET_DIR}/fineweb-edu-sample-10BT-chunked-500-HF4"

# We define our Modal Resources that we'll need
volume = Volume.from_name(VOLUME, create_if_missing=True)
image = Image.debian_slim(python_version="3.9").pip_install(
    "datasets==2.20.0", "huggingface_hub"
)
app = App(image=image)  # Note: prior to April 2024, "app" was called "stub"


# The default timeout is 5 minutes re: https://modal.com/docs/guide/timeouts#handling-timeouts
#  but we override this to
# 6000s to avoid any potential timeout issues
@app.function(
    volumes={DATASET_DIR: volume}, 
    timeout=60000,
    secrets=[Secret.from_name("huggingface-secret")],
)
def upload_dataset(directory, repo):
    import os
    import time

    from huggingface_hub import HfApi
    from datasets import load_from_disk


    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(
        repo_id=repo,
        private=False,
        repo_type="dataset",
        exist_ok=True,
    )

    print("loading from disk")
    dataset=load_from_disk(directory)

    print(f"Pushing to hub {HF_REPO}")
    start = time.perf_counter()
    max_retries = 20
    for attempt in range(max_retries):
        try:
            # api.upload_folder(
            #     folder_path=directory,
            #     repo_id=repo,
            #     repo_type="dataset",
            #     multi_commits=True,
            #     multi_commits_verbose=True,
            # )
            dataset.push_to_hub(repo, num_shards={"train": 99})
            break  # Exit loop if upload is successful
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed, retrying...")
                time.sleep(5)  # Wait for 5 seconds before retrying
            else:
                print("Failed to upload after several attempts.")
                raise  # Re-raise the last exception if all retries fail
    end = time.perf_counter()
    print(f"Uploaded in {end-start}s")


@app.local_entrypoint()
def main():
    upload_dataset.remote(DIRECTORY, HF_REPO)

