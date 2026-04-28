from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import NaturalIdPartitioner
from torch.utils.data import Dataset, DataLoader 
import torchvision.transforms as T


def load_data(partition_id, batch_size):
    #get dataset partition
    fds = FederatedDataset(
        dataset="flwrlabs/femnist",
        partitioners={"train": NaturalIdPartitioner(partition_by="writer_id")}
    )
    partition = fds.load_partition(partition_id=partition_id)

    #describe transform operation on img
    transform = T.Compose([
        T.PILToTensor(),
        T.ConvertImageDtype(float)
    ])

    #apply transform on all PIL in input batch
    def img_to_tensor(batch):
        images = []
        for img in batch["imaged"]:
            images.append(transform(img))
        batch["image"] = images
        return batch

    #apply on partition
    partition = partition.with_transform(img_to_tensor)

    #feed transformed partition in dataloader and return it
    dataloader = DataLoader(
        partition,
        batch_size=batch_size,
        shuffle=True
    )

    return dataloader