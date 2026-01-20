"""Training script."""
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

from src import batch_sampler


def load_nfcorpus(split: str, min_relevance: int) -> tuple[dict, dict, dict]:
    """Load data from NFCorpus.
    
    :param split: train / dev / test
    :return: 3 dictionnaries containing the queries, the docs, and the relevant linked between them.
    """
    dataset = ir_datasets.load(f"nfcorpus/{split}")
    queries = {q.query_id: q.all for q in dataset.queries_iter()}
    docs = {d.doc_id: d.abstract for d in dataset.docs_iter()}
    qrels = {}
    for qrel in dataset.qrels_iter():
        if qrel.relevance >= min_relevance:
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
    ds = Dataset.from_dict({"query": query_texts, "doc": doc_texts})
    return ds


def train(config: dict) -> SentenceTransformer:
    """Perform training."""
    # Fetch data
    query_to_text, doc_to_text, query_to_labels = load_nfcorpus(split="train", min_relevance=config["min_relevance"])
    query_to_text_dev, doc_to_text_dev, query_to_labels_dev = load_nfcorpus(split="dev", min_relevance=config["min_relevance"])

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
    model = SentenceTransformer(config["model"])

    #Training
    args = config["sentence_transformers_args"]
    args["batch_sampler"] = batch_sampler.CustomNoDuplicatesBatchSampler  # @TODO: ideally, this should be a configuration
    trainer = SentenceTransformerTrainer(
        model=model,
        args=SentenceTransformerTrainingArguments(**args),
        train_dataset=ds,
        eval_dataset=ds_dev,
        loss=MultipleNegativesRankingLoss(model),
        evaluator=dev_evaluator,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=config["early_stopping_patience"],
                early_stopping_threshold=config["early_stopping_threshold"],
                )
            ],
        )
    trainer.train()

    return model
