# nfcorpus-finetuning

## Project Overview
Fine-tuning a sentence transformer model for IR on NFCorpus. An example of a fine-tuned model using this repo can be found on Hugging Face [dboyker-code/all-MiniLM-L6-v2-nfcorpus](https://huggingface.co/dboyker-code/all-MiniLM-L6-v2-nfcorpus).

## Installation
The project works with Python 3.12+ and uses [uv](https://docs.astral.sh/uv/). The Dependencies are specified in `pyproject.toml`.

Create a virtual environment.
```
uv sync
```

## Usage
1. Configure the training. The default config is in `./config.yml`. 


2. Train the model.
    ```
    uv run main.py (--config <CONFIG_PATH>)
    ```
    By default, the --config arg points to the default config. If a custom config file is created, its path must be specified here.