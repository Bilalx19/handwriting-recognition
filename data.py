import math
import os
import random

from datasets import load_dataset
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch

DATASET_NAME = "flwrlabs/femnist"
WRITER_ID_COLUMN = "writer_id"
DEFAULT_NUM_PARTITIONS = 4
WRITER_SPLIT_SEED = 42


dataset = None
writer_ids = None
client_splits = {}


def get_dataset():
    global dataset
    if dataset is None:
        dataset = load_dataset(DATASET_NAME, split="train")
    return dataset


def get_writer_ids():
    global writer_ids
    if writer_ids is None:
        ds = get_dataset()
        ids = sorted(set(ds[WRITER_ID_COLUMN]))
        rng = random.Random(WRITER_SPLIT_SEED)
        rng.shuffle(ids)
        writer_ids = ids
    return writer_ids


# describe transform operation on img
_transform = T.Compose([
    T.PILToTensor(),
    T.ConvertImageDtype(torch.float32)
])

# apply transform on all PIL in input batch
def _img_to_tensor(batch):
    images = []
    for img in batch["image"]:
        images.append(_transform(img))
    batch["image"] = images
    return batch


def _writer_ids_for_client(partition_id, num_partitions, writers_per_client=None):
    ids = get_writer_ids()

    if writers_per_client is None:
        writers_per_client = math.ceil(len(ids) / num_partitions)

    start = partition_id * writers_per_client
    end = start + writers_per_client
    selected = ids[start:end]

    if not selected:
        raise ValueError(
            f"No writer IDs assigned to partition {partition_id}. "
            f"num_partitions={num_partitions}, writers_per_client={writers_per_client}, "
            f"available_writers={len(ids)}"
        )

    return selected


def _filter_by_writer_ids(ds, selected_writer_ids):
    selected = set(selected_writer_ids)
    return ds.filter(
        lambda batch: [writer_id in selected for writer_id in batch[WRITER_ID_COLUMN]],
        batched=True,
    )


def train_val_split(partition, val_split=0.2):
    client_writer_ids = sorted(set(partition[WRITER_ID_COLUMN]))

    if len(client_writer_ids) < 2:
        raise ValueError(
            "Writer-disjoint validation needs at least two writer IDs per client. "
            f"Got {len(client_writer_ids)}."
        )

    rng = random.Random(WRITER_SPLIT_SEED)
    rng.shuffle(client_writer_ids)

    num_val_writers = max(1, int(round(len(client_writer_ids) * val_split)))
    num_val_writers = min(num_val_writers, len(client_writer_ids) - 1)

    val_writer_ids = set(client_writer_ids[:num_val_writers])
    train_writer_ids = set(client_writer_ids[num_val_writers:])

    train_partition = _filter_by_writer_ids(partition, train_writer_ids)
    val_partition = _filter_by_writer_ids(partition, val_writer_ids)

    train_partition = train_partition.with_transform(_img_to_tensor)
    val_partition = val_partition.with_transform(_img_to_tensor)

    return train_partition, val_partition


def _load_client_datasets(partition_id, num_partitions, writers_per_client):
    cache_key = (partition_id, num_partitions, writers_per_client)
    if cache_key in client_splits:
        return client_splits[cache_key]

    partition_id = int(partition_id)
    num_partitions = int(num_partitions)

    selected_writer_ids = _writer_ids_for_client(
        partition_id=partition_id,
        num_partitions=num_partitions,
        writers_per_client=writers_per_client,
    )

    partition = _filter_by_writer_ids(get_dataset(), selected_writer_ids)

    train_partition, val_partition = train_val_split(partition)
    client_splits[cache_key] = (train_partition, val_partition)
    return client_splits[cache_key]


def load_data_split(partition_id, batch_size, num_partitions=DEFAULT_NUM_PARTITIONS):
    partition_id = int(partition_id)
    num_partitions = int(num_partitions)

    writers_per_client_env = os.getenv("WRITERS_PER_CLIENT")
    writers_per_client = (
        int(writers_per_client_env) if writers_per_client_env else None
    )

    train_partition, val_partition = _load_client_datasets(
        partition_id=partition_id,
        num_partitions=num_partitions,
        writers_per_client=writers_per_client,
    )

    train_dataloader = DataLoader(train_partition, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_partition, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader
