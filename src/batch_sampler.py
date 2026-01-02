
"""Custom batch sample."""
from collections.abc import Iterator

import torch
from datasets import Dataset
from sentence_transformers.sampler import NoDuplicatesBatchSampler


class CustomNoDuplicatesBatchSampler(NoDuplicatesBatchSampler):
    """Custom batch sampler inspired by Sentence Transformers NoDuplicatesBatchSampler.
    
    Same implementation but allows having several times the same query in the same batch, if the associated docs are different.
    """
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        drop_last: bool,
        valid_label_columns: list[str] | None = None,
        generator: torch.Generator | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__(
            dataset,
            batch_size=batch_size,
            drop_last=drop_last,
            valid_label_columns=valid_label_columns,
            generator=generator,
            seed=seed,
        )

    def __iter__(self) -> Iterator[list[int]]:
        if self.generator and self.seed is not None:
            self.generator.manual_seed(self.seed + self.epoch)

        remaining_indices = dict.fromkeys(torch.randperm(len(self.dataset), generator=self.generator).tolist())
        while remaining_indices:
            batch_values = set()
            batch_indices = []
            for index in remaining_indices:
                sample_values = {str(value) for key, value in self.dataset[index].items() if key == "query"}
                if sample_values & batch_values:
                    continue
                batch_indices.append(index)
                if len(batch_indices) == self.batch_size:
                    yield batch_indices
                    break
                batch_values.update(sample_values)
            else:
                if not self.drop_last:
                    yield batch_indices
            for index in batch_indices:
                del remaining_indices[index]
