# fineweb-modal

This repository is a set of scripts used to process and embed a sample of the [FineWeb-edu dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) using on-demand infrastructure via [Modal](https://modal.com).

The first resulting dataset has been published at [this HuggingFace dataset](https://huggingface.co/datasets/enjalot/fineweb-edu-sample-10BT-chunked-500-nomic-text-v1.5).

All of these scripts have been developed as part of my learning process to scale up my capacity for embedding large datasets. 
As such they aren't immediately generalizable but can be treated as a reference implementation. A lot of it is adapted from the [Embedding Wikipedia](https://modal.com/blog/embedding-wikipedia) tutorial.


## Process

### download.py
To start with, we need to download the HF dataset to a volume in Modal. This is relatively straight forward and easy to change to a different dataset.

### chunker.py
I wanted to pre-chunk my dataset since tokenizing is relatively CPU intensive and my initial experiments with the tutorial code we bottlenecked by the chunking process. I also wanted to use actual token counts and analyze the impact of chunking on the dataset.

I found that the 9.6 million documents in the 10BT sample turned into ~25 million chunks with 10.5 billion tokens due to the 10% overlap I chose. There is an issue in the chunking code right now that I will fix soon where chunks <= 50 tokens are created even though they represent pure overlap and aren't needed.

I based everything on files in the dataset, so the 10BT sample was 99 arrow files, which allowed me to take advantage of Modal's automatic container scaling. Each file is processed by its own container which dramatically sped up the process.

The chunking process took ~40 minutes using 100 containers and cost $5.

### embed-tei.py
This script uses the [Text Embeddings Interface](https://huggingface.co/docs/text-embeddings-inference/en/index) like the wikipedia tutorial, but loading the pre-chunked dataset and creating batches that attempt to fit the batch token limit. So we can pack many more small chunks into a single batch to speed things up.

I believe I'm not quite properly utilizing TEI because I only got ~60% GPU utilization and was only using 10GB memory in the A10G GPUs that have 24GB available. So there is probably a way to speed this up even more. That said it only cost ~$50 to embed the entire dataset. It did take ~12 hours because I didn't always have my full allocation of 10 GPUs available.

### summary.py
I found it useful to quickly calculate summary statistics using the same parallel process of loading each file in its own container and performaing some basic pandas calculations.

### fetch.py
I made a quick utility to download a single file to inspect locally, which was used in the [notebooks/validate.ipynb](notebooks/validate.ipynb) notebook to confirm that the embedding process was working as expected.


## Notebooks
I'm including several notebooks that I developed in the process of learning this in case they are helpful to others.

### [small_sample.ipynb](notebooks/small_sample.ipynb)
The first thing I did was download some very small samples of the dataset and explore them with [Latent Scope](https://github.com/enjalot/latent-scope) to familiarize myself with the data and validate the idea of embedding the dataset.

### [perfile.ipynb](notebooks/perfile.ipynb)
After I struggled with the structure of the wikipedia tutorial I realized I could leverage the CPU parallelism of Modal to process each file in its own container. This notebook was me working out the chunking logic on a single file that I could then parallelize in the `chunker.py` script.

### [validate.ipynb](notebooks/validate.ipynb)
This notebook is me taking a look at a single file that was processed and then trying to understand why I was seeing such small chunks. It led me to realize the mistake I made of keeping around <50 token chunks (which I still need to fix in the chunker.py script...)

## Experimental
On the way to developing this I was trying to understand how to choose batch sizes and token limits. There are two scripts here:

### batchsize.py
This script uses crude measurement techniques to see how much memory gets filled by a batch of tokens. I'm not confident in it anymore because I was able to fit a lot more tokens into the batches I submitted to `embed-tei.py` than I predicted using a A10G instead of an H100.

### embed.py
This script uses the HuggingFace transformers directly (instead of TEI) so I could have a little more control over how I was embedding. It's the same kind of code I use in Latent Scope for locally embedding smaller datasets so it allowed me to better understand the scaling process.
The problem is that it's just much slower than TEI.
