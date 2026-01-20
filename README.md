# nfcorpus-finetuning

## Project Overview
This repo contains the code to fine-tuning a sentence transformer model for IR on NFCorpus. An example of a fine-tuned model can be found on Hugging Face [dboyker-code/all-MiniLM-L6-v2-nfcorpus](https://huggingface.co/dboyker-code/all-MiniLM-L6-v2-nfcorpus).

The training using a *Multiple Negatives Ranking Loss* and a custom batch sampling.

## Installation
The project works with Python 3.12+ and uses [uv](https://docs.astral.sh/uv/). The Dependencies are specified in `pyproject.toml`.

Create a virtual environment.
```
uv sync
```

## Usage
1. Configure the training. The default config is in `./config.yml`. It can be modified or a new custom `.yml` file can be created.


2. Train the model.
    ```
    uv run main.py (--config <CONFIG_PATH>)
    ```
    By default, the --config arg points to the default config `./config.yml`. If a custom config file is created, its path must be specified here.