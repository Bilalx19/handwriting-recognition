from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import NaturalIdPartitioner
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch


fds = None


def get_fds():
    global fds
    if fds is None:
        fds = FederatedDataset(
            dataset="flwrlabs/femnist",
            partitioners={
                "train": NaturalIdPartitioner(partition_by="hsf_id")
            },
        )
    return fds


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


def train_val_split(partition, val_split=0.2, batch_size=32):
    # get number of samples in partition
    num_samples = len(partition)

    # calculate number of samples for validation set
    num_val_samples = int(num_samples * val_split) # 20%
    num_train_samples = num_samples - num_val_samples # 80%

    # split partition into train and validation sets
    partition = partition.train_test_split(test_size=num_val_samples, seed=42)

    # TODO: what does this mean for how many clients we will have?
    # wrap train and validation in dataloaders and return
    # for 3597 writers and 814277 samples, each partition will have roughly 226 samples,
    # meaning for batch size 32, each epoch will have ~6 batches in train and ~2 batches in val
    train_dataloader = DataLoader(partition["train"], batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(partition["test"], batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader

def load_data_split(partition_id, batch_size):
    # load the raw partition for this client
    fds = get_fds()
    partition = fds.load_partition(partition_id=partition_id)

    # apply on partition
    partition = partition.with_transform(_img_to_tensor)

    # delegate splitting and dataloader creation to train_val_split
    return train_val_split(partition, batch_size=batch_size)