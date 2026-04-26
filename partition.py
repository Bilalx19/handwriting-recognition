from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import NaturalIdPartitioner

fds = FederatedDataset(
    dataset="flwrlabs/femnist",
    partitioners={"train": NaturalIdPartitioner(partition_by="writer_id")}
)
partition = fds.load_partition(partition_id=0)

partition.save_to_disk("./datasets/partition_0")