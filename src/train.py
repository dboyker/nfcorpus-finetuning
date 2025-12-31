"""Training script."""
import random

import ir_datasets
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.losses import MultipleNegativesRankingLoss
from transformers import EarlyStoppingCallback

import batch_sampler

SEED = 42
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_ARGS = dict(
    output_dir="models/test",
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
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_THRESHOLD = 0.01
MIN_RELEVANCE = 2
random.seed(SEED)


def load_nfcorpus(split: str) -> tuple[dict, dict, dict]:
    """Load data from NFCorpus.
    
    :param split: train / dev / test
    :return: 3 dictionnaries containing the queries, the docs, and the relevant linked between them.
    """
    dataset = ir_datasets.load(f"nfcorpus/{split}")
    queries = {q.query_id: q.all for q in dataset.queries_iter()}
    docs = {d.doc_id: d.abstract for d in dataset.docs_iter()}
    qrels = {}
    for qrel in dataset.qrels_iter():
        if qrel.relevance >= MIN_RELEVANCE:
            qrels.setdefault(qrel.query_id, []).append(qrel.doc_id)
    return queries, docs, qrels


def build_dataset(queries, docs, qrels) -> Dataset:
    """Build a HF dataset.

    @TODO: the dataset includes the same queries several times. This could mess up with MultipleNegativesRankingLoss.
    
    :param queries: a dict containing the query ids and their corresponding text
    :param docs: a dict containing the doc ids and their corresponding text
    :param qrels: a dict containing the query ids and their relevant doc ids
    """
    query_texts = []
    doc_texts = []
    for qid, doc_ids in qrels.items():
            q_text = queries.get(qid)
            if not q_text:
                continue
            for doc_id in doc_ids:
                d_text = docs.get(doc_id)
                if d_text:
                    query_texts.append(q_text)
                    doc_texts.append(d_text)
    ds = Dataset.from_dict({"query": query_texts, "doc": doc_texts}).shuffle(seed=SEED)
    return ds


def main(override_args={}) -> SentenceTransformer:
    """Perform training."""
    # Fetch data
    if override_args is None:
        override_args = {}
    query_to_text, doc_to_text, query_to_labels = load_nfcorpus(split="train")
    query_to_text_dev, doc_to_text_dev, query_to_labels_dev = load_nfcorpus(split="dev")

    # Build datasets
    ds = build_dataset(query_to_text, doc_to_text, query_to_labels)
    ds_dev = build_dataset(query_to_text_dev, doc_to_text_dev, query_to_labels_dev)
    
    # Evaluator
    dev_evaluator = InformationRetrievalEvaluator(
        queries=query_to_text_dev,
        corpus=doc_to_text_dev,
        relevant_docs=query_to_labels_dev,
        name="dev",
    )

    # Model
    model = SentenceTransformer(MODEL)

    #Training
    args = DEFAULT_ARGS | override_args
    trainer = SentenceTransformerTrainer(
        model=model,
        args=SentenceTransformerTrainingArguments(**args),
        train_dataset=ds,
        eval_dataset=ds_dev,
        loss=MultipleNegativesRankingLoss(model),
        evaluator=dev_evaluator,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=EARLY_STOPPING_PATIENCE,
                early_stopping_threshold=EARLY_STOPPING_THRESHOLD,
                )
            ],
        )
    trainer.train()

    return model


if __name__ == "__main__":
    main()