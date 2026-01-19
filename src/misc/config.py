import random

import yaml

from src import sentence_transformers_args


def load_config(path: str):
    """Load a yml config based on a given path."""
    with open(path) as file:
        config = yaml.safe_load(file)
    config["sentence_transformers_args"] = getattr(sentence_transformers_args, config["sentence_transformers_args"])
    random.seed(config["seed"])
    return config