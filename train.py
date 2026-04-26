from datasets import load_from_disk
from partition import partition


if __name__ == "__main__":
    dataset_partition = load_from_disk("./datasets/partition_0")
    print(dataset_partition)