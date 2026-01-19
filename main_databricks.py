# Databricks notebook source
# MAGIC %pip install ./
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC from src.train import train
# MAGIC from src.misc.config import load_config

# COMMAND ----------

path = "./config.yml"
config = load_config(path)
model_name = "all-MiniLM-L6-v2-nfcorpus"
config["sentence_transformers_args"]["output_dir"] = f"fine_tuned_models/{model_name}"
model = train(config=config)
model.save(f"/dbfs/saved_models/{model_name}")

