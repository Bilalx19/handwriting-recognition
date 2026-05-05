from data import load_data
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import NaturalIdPartitioner


dataloader = load_data(partition_id=0, batch_size=32)

batch = next(iter(dataloader))

images = batch["image"]
true_char = batch["character"]

print(images.shape)
print(true_char.shape)
print(true_char[:10])

fds = FederatedDataset(
    dataset="flwrlabs/femnist",
    partitioners={"train": NaturalIdPartitioner(partition_by="writer_id")}
)

print(fds.partitioners["train"].num_partitions)