"""
Customized trainer for setnece highlight
"""
# import time
# import json
# import collections
# import multiprocessing
import torch
from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union
from torch.utils.data import Sampler
from transformers import Trainer
from transformers.trainer_utils import has_length

_is_torch_generator_available = False

class AlbertTrainer(Trainer):

    def inference(self, eval_dataset=None):
        for b, batch in enumerate(eval_dataset):
            for k in batch:
                batch[k] = batch[k].to(self.args.device)
            output = self.model.inference(batch)
            logit = output['logit']


class AlbertTrainerForConvBatch(Trainer):

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:

        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        generator = None
        if self.args.world_size <= 1 and _is_torch_generator_available:
            generator = torch.Generator()
            # for backwards compatibility, we generate a seed here (which is sampled from a generator seeded with
            # `args.seed`) if data_seed isn't provided.
            # Further on in this method, we default to `args.seed` instead.
            if self.args.data_seed is None:
                seed = int(torch.empty((), dtype=torch.int64).random_().item())
            else:
                seed = self.args.data_seed
            generator.manual_seed(seed)

        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed

        # Build the sampler.
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            if self.args.world_size <= 1:
                return LengthGroupedSampler(
                    self.args.train_batch_size * self.args.gradient_accumulation_steps,
                    dataset=self.train_dataset,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    generator=generator,
                )
            else:
                return DistributedLengthGroupedSampler(
                    self.args.train_batch_size * self.args.gradient_accumulation_steps,
                    dataset=self.train_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    seed=seed,
                )

        else:
            if self.args.world_size <= 1:
                if _is_torch_generator_available:
                    # return RandomSampler(self.train_dataset, generator=generator)
                    return RandomBatchSequetialSampler(
                            self.train_dataset, 
                            generator=generator, 
                            batch_size=self.args.per_device_train_batch_size
                    )
                return RandomBatchSequetialSampler(self.train_dataset, batch_size=self.args.per_device_train_batch_size)
            elif (
                self.args.parallel_mode in [ParallelMode.TPU, ParallelMode.SAGEMAKER_MODEL_PARALLEL]
                and not self.args.dataloader_drop_last
            ):
                # Use a loop for TPUs when drop_last is False to have all batches have the same size.
                return DistributedSamplerWithLoop(
                    self.train_dataset,
                    batch_size=self.args.per_device_train_batch_size,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=seed,
                )
            else:
                return DistributedSampler(
                    self.train_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=seed,
                )

class RandomBatchSequetialSampler(Sampler[int]):
    r"""Yeild the random batch, which are composed of the ordered(sequetial) examples within each batch. """
    data_source: Sized
    replacement: bool

    def __init__(self, data_source: Sized, replacement: bool = False,
                 num_samples: Optional[int] = None, generator=None, batch_size=8) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator
        self.batch_size = batch_size

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        # RandomSampler
        # return iter(range(n))
        n = len(self.data_source)
        batch_list = [
                list(range(0, n))[s:e] for s, e in zip(
                    range(0, n, self.batch_size), range(self.batch_size, n+self.batch_size, self.batch_size)
                )
        ]
        rbss_list = list()

        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        # permutation
        for i in torch.randperm(len(batch_list), generator=generator).tolist():
            rbss_list += batch_list[i]

        return iter(rbss_list)

    def __len__(self) -> int:
        return self.num_samples

