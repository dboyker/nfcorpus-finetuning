import yaml


def load_config(path: str):
    """Load a yml config based on a given path."""
    with open(path) as file:
        config = yaml.safe_load(file)
    return config