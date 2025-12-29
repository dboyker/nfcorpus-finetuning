"""Training script."""
import ir_datasets
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.losses import MultipleNegativesRankingLoss

SEED = 42
MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
ARGS = SentenceTransformerTrainingArguments(
    output_dir="models/test",
    num_train_epochs=3,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    warmup_ratio=0.1,
    learning_rate=3e-5,
    weight_decay=0.01,
    fp16=True,
    bf16=False,
    eval_strategy="steps",
    eval_steps=100,
    logging_steps=100,
    disable_tqdm=True,
)


def load_data(split) -> tuple[dict, dict, dict]:
    """Load data from NFCorpus.
    
    :param split: train / dev / test
    :return: 3 dictionnaries containing the queries, the docs, and the relevant linked between them.
    """
    dataset = ir_datasets.load(f"nfcorpus/{split}")
    query_to_text = {q.query_id: q.all for q in dataset.queries_iter()}
    doc_to_text = {d.doc_id: d.abstract for d in dataset.docs_iter()}
    query_to_labels = {}
    used_docs = []
    for qrel in dataset.qrels_iter():
        if qrel.relevance >= 2:
            if qrel.doc_id not in used_docs:
                used_docs.append(qrel.doc_id)
            query_to_labels.setdefault(qrel.query_id, []).append(qrel.doc_id)
    """
    print(len(query_to_text.keys()))
    print(len(doc_to_text.keys()))
    print(len(query_to_labels))
    print(len(used_docs))
    """
    return query_to_text, doc_to_text, query_to_labels


def build_dataset(query_to_text, doc_to_text, query_to_labels) -> Dataset:
    """Build a HF dataset.
    
    :param query_to_text: a dict containing the query ids and their corresponding text
    :param doc_to_text: a dict containing the doc ids and their corresponding text
    :param query_to_labels: a dict containing the query ids and their relevant doc ids
    """
    queries = [query_to_text[q] for q, docs in query_to_labels.items() for _ in docs]
    docs = [doc_to_text[d] for _, docs in query_to_labels.items() for d in docs]
    ds = Dataset.from_dict({"queries": queries, "docs": docs}).shuffle(seed=SEED)
    return ds


def main() -> None:
    """Perform training."""
    # Fetch data
    query_to_text, doc_to_text, query_to_labels = load_data(split="train")
    query_to_text_dev, doc_to_text_dev, query_to_labels_dev = load_data(split="dev")

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

    #Training
    trainer = SentenceTransformerTrainer(
        model=MODEL,
        args=ARGS,
        train_dataset=ds,
        eval_dataset=ds_dev,
        loss=MultipleNegativesRankingLoss(MODEL),
        evaluator=dev_evaluator,
        )
    trainer.train()


if __name__ == "__main__":
    main()