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
print(config)
model_name = "all-MiniLM-L6-v2-nfcorpus"
model = train(config=config, override_sentence_transformers_args=dict(output_dir=f"fine_tuned_models/{model_name}"))
model.save(f"/dbfs/saved_models/{model_name}")
