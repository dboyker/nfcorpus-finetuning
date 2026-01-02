# Databricks notebook source
# MAGIC %pip install ./
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC from src import train

# COMMAND ----------

model_name = "all-MiniLM-L6-v2-nfcorpus"
model = train.main(override_args=dict(output_dir=f"fine_tuned_models/{model_name}"))
model.save(f"/dbfs/saved_models/{model_name}")

# COMMAND ----------

