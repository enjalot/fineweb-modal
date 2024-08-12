from modal import App, Image, Volume

NUM_CPU=16
MAX_TOKENS = 500
OVERLAP = 0.1 # 10% overlap when chunking

# We first set out configuration variables for our script.
DATASET_DIR = "/data"
DATASET_NAME = "HuggingFaceFW/fineweb-edu"
# DATASET_FILES = "sample/10BT/*.parquet"
DATASET_SAVE ="fineweb-edu-sample-100BT"
DATASET_SAVE_CHUNKED = f"fineweb-edu-sample-100BT-chunked-{MAX_TOKENS}"

# MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"

# We define our Modal Resources that we'll need
volume = Volume.from_name("embedding-fineweb-edu", create_if_missing=True)
image = Image.debian_slim(python_version="3.9").pip_install(
    "datasets==2.16.1", "apache_beam==2.53.0", "transformers", "pandas", "tqdm"
)
app = App(image=image)  # Note: prior to April 2024, "app" was called "stub"

def chunk_row(row, tokenizer):
    # print("ROW", row)
    keep_keys = ["id", "url", "score", "dump"]
    text = row["text"]
    chunks = []

    tokens = tokenizer.encode(text)
    token_count = len(tokens)
    if token_count > MAX_TOKENS:
        overlap = int(MAX_TOKENS * OVERLAP)
        start_index = 0
        ci = 0
        while start_index < len(tokens):
            end_index = min(start_index + MAX_TOKENS, len(tokens))
            chunk = tokens[start_index:end_index]
            if len(chunk) < overlap * MAX_TOKENS:
                break
            chunks.append({
                "chunk_index": ci,
                "chunk_text": tokenizer.decode(chunk),
                "chunk_tokens": chunk,
                "chunk_token_count": len(chunk),
                **{key: row[key] for key in keep_keys}
            })
            start_index += MAX_TOKENS - overlap
            ci += 1
    else:
        chunks.append({
            "chunk_index": 0,
            "chunk_text": text,
            "chunk_tokens": tokens,
            "chunk_token_count": token_count,
            **{key: row[key] for key in keep_keys}
        })

    return chunks


@app.function(cpu=NUM_CPU, volumes={DATASET_DIR: volume}, timeout=3000)
def process_dataset(file):
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm
    import pandas as pd
    import transformers
    transformers.logging.set_verbosity_error()
    from transformers import AutoTokenizer
    from datasets import load_from_disk, load_dataset
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", model_max_length=MAX_TOKENS)

    start = time.perf_counter()
    # Load the dataset as a Hugging Face dataset
    print(f"Loading dataset from {DATASET_DIR}/{DATASET_SAVE}/train/{file}")
    dataset = load_dataset("arrow", data_files=f"{DATASET_DIR}/{DATASET_SAVE}/train/{file}")
    df = pd.DataFrame(dataset['train'])
    print("dataset", len(df))
    print(f"Dataset loaded in {time.perf_counter()-start:.2f} seconds for {file}") 

    chunks_list = []
    with ThreadPoolExecutor(max_workers=NUM_CPU) as executor:
        pbar = tqdm(total=len(df), desc=f"Processing Rows for {file}")
        
        # this gets called inside each thread
        def process_batch(batch):
            batch_chunks = []
            for row in batch:
                row_chunks = chunk_row(row, tokenizer)
                pbar.update(1)
                batch_chunks.extend(row_chunks)
            return batch_chunks

        print(f"making batches for {file}")
        batch_size = 200  # Adjust batch size based on your needs
        batches = [df.iloc[i:i + batch_size].to_dict(orient="records") for i in range(0, len(df), batch_size)]
        print(f"made batches for {file}")
        print(f"setting up futures for {file}")
        futures = [executor.submit(process_batch, batch) for batch in batches]
        # futures = [executor.submit(chunk_row, row) for index, row in df.iterrows()]
        # for future in tqdm(as_completed(futures), total=len(df), desc="Processing Rows"):
        #     chunks_list.extend(future.result())
        print(f"in the future for {file}")
        # pbar = tqdm(total=len(df)//batch_size, desc="Processing Rows")
        for future in as_completed(futures):
            chunks_list.extend(future.result())
            # print(len(chunks_list))
            # pbar.update(1)  # Manually update the progress bar
        pbar.close()

    chunked_df = pd.DataFrame(chunks_list)
    file_name = file.split(".")[0]
    import os
    output_dir = f"{DATASET_DIR}/{DATASET_SAVE_CHUNKED}/train"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"saving to {output_dir}/{file_name}.parquet")
    chunked_df.to_parquet(f"{output_dir}/{file_name}.parquet")
    print(f"done with {file}, {len(chunks_list)} chunks")
    volume.commit()
    return f"All done with {file}", len(chunks_list)


@app.local_entrypoint()
def main():
    # download_dataset.remote()
    # from huggingface_hub import HfFileSystem
    # hffs = HfFileSystem()
    # files = hffs.ls("datasets/HuggingFaceFW/fineweb-edu/sample/10BT", detail=False)

    files = [f"data-{i:05d}-of-00989.arrow" for i in range(989)]
    
    # process_dataset.remote(file, max_tokens=MAX_TOKENS, num_cpu=NUM_CPU)
    for resp in process_dataset.map(files, order_outputs=False, return_exceptions=True):
        if isinstance(resp, Exception):
            print(f"Exception: {resp}")
            continue
        print(resp)


