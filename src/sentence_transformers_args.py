from src import batch_sampler

DEFAULT_ARGS = dict(
    output_dir="fine_tuned_models/default",
    num_train_epochs=5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    warmup_ratio=0.1,
    learning_rate=3e-5,
    weight_decay=0.01,
    fp16=True,
    bf16=False,
    batch_sampler=batch_sampler.CustomNoDuplicatesBatchSampler,
    metric_for_best_model="eval_loss",
    eval_strategy="steps",
    eval_steps=100,
    logging_steps=100,
    disable_tqdm=True,
    load_best_model_at_end=True,
    )

CUSTOM_ARGS = DEFAULT_ARGS | dict(output_dir="fine_tuned_models/all-MiniLM-L6-v2-nfcorpus")