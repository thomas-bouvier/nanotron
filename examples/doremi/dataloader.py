import dataclasses
import math
import warnings
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from doremi_context import DoReMiContext
from nanotron import distributed as dist
from nanotron import logging
from nanotron.config import (
    PretrainDatasetsArgs,
)
from nanotron.dataloader import EmptyInfiniteDataset, SkipBatchSampler, clm_process, get_dataloader_worker_init
from nanotron.logging import log_rank
from nanotron.parallel import ParallelContext
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer
from nanotron.parallel.pipeline_parallel.utils import get_input_output_pp_ranks
from nanotron.trainer import DistributedTrainer
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

try:
    from datasets import Dataset, DatasetDict, load_dataset
    from huggingface_hub import __version__ as hf_hub_version
    from transformers import AutoTokenizer
    from transformers import __version__ as tf_version
    from transformers.trainer_pt_utils import DistributedSamplerWithLoop
except ImportError:
    warnings.warn("Datasets and/or Transformers not installed, you'll be unable to use the dataloader.")


logger = logging.get_logger(__name__)


def get_doremi_datasets(
    hf_dataset: str,
    domain_keys: List[str],
    splits: Optional[Union[List[str], str]] = ["train", "test"],
) -> List[DatasetDict]:
    if isinstance(splits, str):
        splits = [splits]

    raw_datasets = DatasetDict()
    for split in splits:
        raw_datasets[split] = []
        for domain_key in domain_keys:
            d = load_dataset(
                hf_dataset,
                domain_key,
                split=split,
            )
            raw_datasets[split].append(d)

    return raw_datasets


def get_dataloader(trainer: DistributedTrainer, domain_keys: List[str]):
    """Returns a dataloader for training."""
    assert isinstance(trainer.config.data.dataset, PretrainDatasetsArgs), "Please provide a dataset in the config file"

    log_rank("Using `datasets` library", logger=logger, level=logging.INFO, rank=0)

    tokenizer_path = trainer.config.tokenizer.tokenizer_name_or_path
    log_rank(
        f"Loading tokenizer from {tokenizer_path} and transformers/hf_hub versions {tf_version, hf_hub_version}",
        logger=logger,
        level=logging.INFO,
        rank=0,
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    log_rank(
        f"Downloading datasets from {trainer.config.data.dataset.hf_dataset_or_datasets}",
        logger=logger,
        level=logging.INFO,
        rank=0,
    )

    raw_datasets = get_doremi_datasets(
        hf_dataset=trainer.config.data.dataset.hf_dataset_or_datasets,
        domain_keys=domain_keys,
        splits=trainer.config.data.dataset.hf_dataset_splits,
    )["train"]

    train_datasets = []
    for raw_dataset in raw_datasets:
        train_datasets.append(
            clm_process(
                raw_dataset=raw_dataset,
                tokenizer=tokenizer,
                text_column_name=trainer.config.data.dataset.text_column_name,
                dataset_processing_num_proc_per_process=trainer.config.data.dataset.dataset_processing_num_proc_per_process,
                dataset_overwrite_cache=trainer.config.data.dataset.dataset_overwrite_cache,
                sequence_length=trainer.sequence_length,
            )
        )

    # NOTE: We load the processed dataset on the ranks requiring it
    input_pp_rank, output_pp_rank = get_input_output_pp_ranks(model=trainer.model)
    doremi_context = trainer.doremi_context
    dataloader = get_doremi_dataloader(
        doremi_context=doremi_context,
        train_datasets=train_datasets,
        ref_model=trainer.ref_model,
        sequence_length=trainer.sequence_length,
        parallel_context=trainer.parallel_context,
        input_pp_rank=input_pp_rank,
        output_pp_rank=output_pp_rank,
        micro_batch_size=trainer.micro_batch_size,
        consumed_train_samples=trainer.consumed_train_samples,
        dataloader_num_workers=trainer.config.data.num_loading_workers,
        seed_worker=trainer.config.data.seed,
        dataloader_drop_last=True,
    )()

    # Check if we have enough samples for train_steps
    # assert (
    #     trainer.config.tokens.train_steps - trainer.start_iteration_step
    # ) * trainer.global_batch_size // trainer.parallel_context.dp_pg.size() < len(dataloader), (
    #     f"Dataset is too small for steps ({len(dataloader)} < {(trainer.config.tokens.train_steps - trainer.start_iteration_step) * trainer.global_batch_size // trainer.parallel_context.dp_pg.size()}), "
    #     f"Try train_steps<={len(dataloader) * trainer.parallel_context.dp_pg.size() // trainer.global_batch_size + trainer.start_iteration_step}"
    # )
    # else:
    #     raise ValueError(f"Unhandled case of `self.config.data.dataset`. Got: {trainer.config.data.dataset}")

    return dataloader


@dataclasses.dataclass
class DataCollatorForCLM:
    """
    Data collator used for causal language modeling.

    - input_pp_rank: Discards last input id token
    - output_pp_rank: Discards first label id token
    - other pp ranks: Don't have data. Instead, we use `TensorPointer` to point to the rank having the data.
    """

    sequence_length: int
    input_pp_rank: int
    output_pp_rank: int
    parallel_context: ParallelContext

    def __call__(self, examples: List[Dict[str, List[np.ndarray]]]) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        # Process the case when current rank doesn't require data. We return `TensorPointer` that points to ranks having the data.
        current_pp_rank = dist.get_rank(self.parallel_context.pp_pg)
        if current_pp_rank not in [
            self.input_pp_rank,
            self.output_pp_rank,
        ]:
            assert all(len(example) == 0 for example in examples)
            return {
                "input_ids": TensorPointer(self.input_pp_rank),
                "input_mask": TensorPointer(self.input_pp_rank),
                "label_ids": TensorPointer(self.output_pp_rank),
                "label_mask": TensorPointer(self.output_pp_rank),
            }

        assert all(list(example.keys()) == ["input_ids", "domain_ids"] for example in examples)

        # TODO @nouamanetazi: Is it better to have examples as np.array or torch.Tensor?
        input_ids = np.vstack([examples[i]["input_ids"] for i in range(len(examples))])  # (b, s)
        batch_size, expanded_input_length = input_ids.shape

        result: Dict[str, Union[np.ndarray, TensorPointer]] = {}

        result["input_ids"] = TensorPointer(group_rank=self.input_pp_rank)
        result["input_mask"] = TensorPointer(group_rank=self.input_pp_rank)
        result["label_ids"] = TensorPointer(group_rank=self.output_pp_rank)
        result["label_mask"] = TensorPointer(group_rank=self.output_pp_rank)

        assert (
            expanded_input_length == self.sequence_length + 1
        ), f"Samples should be of length {self.sequence_length + 1} (seq_len+1), but got {expanded_input_length}"

        # Process inputs: last token is the label
        if current_pp_rank == self.input_pp_rank:
            result["input_ids"] = input_ids[:, :-1]
            result["input_mask"] = np.ones((batch_size, self.sequence_length), dtype=np.bool_)

        # Process labels: shift them to the left
        if current_pp_rank == self.output_pp_rank:
            result["label_ids"] = input_ids[:, 1:]
            result["label_mask"] = np.ones((batch_size, self.sequence_length), dtype=np.bool_)
            # NOTE: only the last pipeline stage needs domain_idxs for computing DoReMi loss
            result["domain_idxs"] = np.vstack([examples[i]["domain_ids"] for i in range(len(examples))])

        if isinstance(result["input_ids"], torch.Tensor) and result["input_ids"].shape[-1] != self.sequence_length:
            raise ValueError(
                f"`labels` are incorrectly preprocessed. `labels` length is {result['input_ids'].shape[-1]}, but should be"
                f" {self.sequence_length}."
            )
        if isinstance(result["label_ids"], torch.Tensor) and result["label_ids"].shape[-1] != self.sequence_length:
            raise ValueError(
                f"`labels` are incorrectly preprocessed. `labels` length is {result['label_ids'].shape[-1]}, but should be"
                f" {self.sequence_length}."
            )

        # Cast np.array to torch.Tensor
        result = {k: v if isinstance(v, TensorPointer) else torch.from_numpy(v) for k, v in result.items()}
        return result


class DistributedSamplerForDoReMi(DistributedSampler):
    def __init__(
        self,
        datasets: List[Dataset],
        batch_size: int,
        doremi_context: DoReMiContext,
        parallel_context: ParallelContext,
        **kwargs,
    ):
        super().__init__(datasets, **kwargs)
        self.datasets = datasets
        self.batch_size = batch_size
        self.domain_weights = doremi_context.domain_weights
        self.total_size = self._calculate_total_size()
        self.parallel_context = parallel_context

        # Random generator
        generator = torch.Generator(device="cpu")
        # Make sure that TP are synced always
        # TODO(xrsrke): make seed configurable
        seed = 42
        self.generator = generator.manual_seed(
            seed * (1 + dist.get_rank(self.parallel_context.dp_pg)) * (1 + dist.get_rank(self.parallel_context.pp_pg))
        )

    def _calculate_total_size(self):
        total_samples = sum(len(d) for d in self.datasets)
        # total_samples = sum(compute_total_sample_per_streaming_dataset(self.datasets))
        return math.ceil(total_samples / self.batch_size) * self.batch_size

    def __iter__(self):
        domain_indices = []

        lengths = [len(d) for d in self.datasets]
        # lengths = compute_total_sample_per_streaming_dataset(self.datasets)

        offsets = np.cumsum([0] + lengths[:-1])

        for i, dataset in enumerate(self.datasets):
            dataset_partition_size = len(dataset) // self.num_replicas
            dataset_partition_offsets = self.rank * dataset_partition_size
            num_samples = int(dataset_partition_size * self.domain_weights[i].item())

            local_indices = (
                torch.randint(
                    low=0, high=dataset_partition_size, size=(num_samples,), generator=self.generator, device="cpu"
                )
                + dataset_partition_offsets
            )
            # NOTE: align the indicies across the combined dataset
            global_indices = local_indices + offsets[i]
            domain_indices.extend(global_indices)

        np.random.shuffle(domain_indices)
        domain_indices = domain_indices[: self.total_size]

        # Yield indices in batches
        for i in range(0, len(domain_indices), self.batch_size):
            yield domain_indices[i : i + self.batch_size]


# Adapted from https://github.com/huggingface/transformers/blob/47e1676255e5dd86b9541f734cd4f4bdcbb50f4a/src/transformers/trainer.py#L763-L835
def _get_train_sampler(
    dp_size: int,
    dp_rank: int,
    train_datasets: "Dataset",
    seed: int,
    use_loop_to_round_batch_size: bool,
    consumed_train_samples: int,
    doremi_context: DoReMiContext,
    parallel_context: ParallelContext,
    micro_batch_size: Optional[int] = None,
    drop_last: Optional[bool] = True,
) -> Optional[torch.utils.data.Sampler]:
    """returns sampler that restricts data loading to a subset of the dataset proper to the DP rank"""

    # Build the sampler.
    # TODO @nouamanetazi: Support group_by_length: https://github.com/huggingface/transformers/blob/47e1676255e5dd86b9541f734cd4f4bdcbb50f4a/src/transformers/trainer.py#L783-L810

    if use_loop_to_round_batch_size:
        assert micro_batch_size is not None
        # loops at the end back to the beginning of the shuffled samples to make each process have a round multiple of batch_size samples.
        sampler = DistributedSamplerWithLoop(
            train_datasets,
            batch_size=micro_batch_size,
            num_replicas=dp_size,
            rank=dp_rank,
            seed=seed,
            drop_last=drop_last,
        )
    else:
        # sampler = DistributedSampler(train_dataset, num_replicas=dp_size, rank=dp_rank, seed=seed, drop_last=drop_last)
        sampler = DistributedSamplerForDoReMi(
            train_datasets,
            batch_size=micro_batch_size,
            num_replicas=dp_size,
            rank=dp_rank,
            seed=seed,
            drop_last=drop_last,
            doremi_context=doremi_context,
            parallel_context=parallel_context,
        )

    if consumed_train_samples > 0:
        sampler = SkipBatchSampler(sampler, skip_batches=consumed_train_samples, dp_size=dp_size)

    return sampler


def compute_total_sample_per_streaming_dataset(datasets: List[Dataset]) -> List[int]:
    lengths = []
    for d in datasets:
        sample_count = 0
        for _ in d:
            sample_count += 1
        lengths.append(sample_count)
    return lengths


class CombinedDataset(Dataset):
    def __init__(self, datasets: List[Dataset]):
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]

        # self.lengths = compute_total_sample_per_streaming_dataset(datasets)
        self.offsets = np.cumsum([0] + self.lengths[:-1])

    def __len__(self) -> int:
        return sum(self.lengths)

    def __getitem__(self, batch_global_idxs: List[List[int]]) -> Dict:
        def merge_outputs(outputs):
            merged_input_ids = sum((o["input_ids"] for o in outputs), [])
            merged_domain_idx = sum((o["domain_idx"] for o in outputs), [])
            return {"input_ids": merged_input_ids, "domain_ids": merged_domain_idx}

        outputs = []
        for global_idxs in batch_global_idxs:
            output = [self._get_sample(global_idx) for global_idx in global_idxs]
            # TODO(xrsrke): refactor this, make it fast
            output = {key: [d[key] for d in output] for key in output[0]}
            outputs.append(output)

        return merge_outputs(outputs)

    def _get_sample(self, global_idx):
        dataset_idx, local_idx = self._get_dataset_and_local_index(global_idx)
        dataset = self.datasets[dataset_idx]
        sample = {key: dataset[key][local_idx] for key in dataset.features}
        sample["domain_idx"] = dataset_idx
        return sample

    def _get_dataset_and_local_index(self, global_idx):
        for i, offset in enumerate(self.offsets):
            if global_idx < offset + self.lengths[i]:
                return i, global_idx - offset

        raise IndexError(f"Index out of range, global_idx={global_idx}")


# Adapted from https://github.com/huggingface/transformers/blob/47e1676255e5dd86b9541f734cd4f4bdcbb50f4a/src/transformers/trainer.py#L837
def get_doremi_dataloader(
    doremi_context: DoReMiContext,
    ref_model: nn.Module,
    train_datasets: List["Dataset"],
    sequence_length: int,
    parallel_context: ParallelContext,
    input_pp_rank: int,
    output_pp_rank: int,
    micro_batch_size: int,
    consumed_train_samples: int,
    dataloader_num_workers: int,
    seed_worker: int,
    dataloader_drop_last: bool = True,
    dataloader_pin_memory: bool = True,
    use_loop_to_round_batch_size: bool = False,
) -> DataLoader:
    # if not isinstance(train_dataset, datasets.Dataset):
    #     raise ValueError(f"training requires a datasets.Dataset, but got {type(train_dataset)}")

    # Case of ranks requiring data
    if dist.get_rank(parallel_context.pp_pg) in [
        input_pp_rank,
        output_pp_rank,
    ]:
        train_datasets = [
            d.with_format(type="numpy", columns=["input_ids"], output_all_columns=True) for d in train_datasets
        ]

    # Case of ranks not requiring data. We give them an infinite dummy dataloader
    else:
        # TODO(xrsrke): recheck this
        # train_datasets = train_datasets[0]
        # assert train_dataset.column_names == ["input_ids"], (
        #     f"Dataset has to have a single column, with `input_ids` as the column name. "
        #     f"Current dataset: {train_dataset}"
        # )
        dataset_length = len(train_datasets[0])
        train_dataset = train_datasets[0].remove_columns(column_names="input_ids")
        assert (
            len(train_dataset) == 0
        ), f"Dataset has to be empty after removing the `input_ids` column. Current dataset: {train_dataset}"
        # HACK as if we remove the last column of a train_dataset, it becomes empty and it's number of rows becomes empty.
        train_datasets = EmptyInfiniteDataset(length=dataset_length)
        # No need to spawn a lot of workers, we can just use main
        dataloader_num_workers = 0

    data_collator = DataCollatorForCLM(
        sequence_length=sequence_length,
        input_pp_rank=input_pp_rank,
        output_pp_rank=output_pp_rank,
        parallel_context=parallel_context,
    )
    log_rank(
        f"Before _get_train_sampler, global_rank={dist.get_rank(parallel_context.world_pg)}",
        logger=logger,
        level=logging.INFO,
    )

    train_sampler = _get_train_sampler(
        dp_size=parallel_context.dp_pg.size(),
        dp_rank=dist.get_rank(parallel_context.dp_pg),
        train_datasets=train_datasets,
        seed=seed_worker,
        use_loop_to_round_batch_size=use_loop_to_round_batch_size,
        micro_batch_size=micro_batch_size,
        drop_last=dataloader_drop_last,
        consumed_train_samples=consumed_train_samples,
        doremi_context=doremi_context,
        parallel_context=parallel_context,
    )

    log_rank(
        f"Before CombinedDataset, global_rank={dist.get_rank(parallel_context.world_pg)}",
        logger=logger,
        level=logging.INFO,
    )

    comebined_dataset = CombinedDataset(train_datasets)

    log_rank(
        f"Before DataLoader, global_rank={dist.get_rank(parallel_context.world_pg)}",
        logger=logger,
        level=logging.INFO,
    )

    dataloader = DataLoader(
        comebined_dataset,
        batch_size=micro_batch_size,
        sampler=train_sampler,
        collate_fn=data_collator,
        drop_last=dataloader_drop_last,  # we also drop_last in `clm_process()`
        num_workers=dataloader_num_workers,
        pin_memory=dataloader_pin_memory,
        worker_init_fn=get_dataloader_worker_init(dp_rank=dist.get_rank(parallel_context.dp_pg)),
    )

    def _data_generator():
        dist.barrier()
        for batch in dataloader:
            batch = {k: v.to("cuda") for k, v in batch.items()}
            log_rank(
                f"Before reference model do inference, global_rank={dist.get_rank(parallel_context.world_pg)}",
                logger=logger,
                level=logging.INFO,
                rank=None,
            )

            # NOTE: because the inference model don't take `domain_idxs` as input
            # we need to remove it from the batch
            batch_for_inference = {k: v for k, v in batch.items() if k != "domain_idxs"}
            ref_losses = ref_model(**batch_for_inference)["losses"]
            batch["ref_losses"] = ref_losses
            yield batch

    return _data_generator
