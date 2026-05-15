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

# ÄNDERUNG 1: Data Augmentation für Training 
# Transform für Training (mit Augmentation)
_transform_train = T.Compose([
    T.RandomRotation(degrees=10),                    # leichte Rotation (+-10°)
    T.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # leichte Verschiwbung
    T.RandomResizedCrop(size=(28, 28), scale=(0.9, 1.0)),  # leichter Zoom/Crop
    T.PILToTensor(),
    T.ConvertImageDtype(torch.float32),
])

# Transform für Validation/Test (nur Standard)
_transform_val = T.Compose([
    T.PILToTensor(),
    T.ConvertImageDtype(torch.float32),
])

# ÄNDERUNG 2: Zwei separate Transform-Funktionen ==========
def _img_to_tensor_train(batch):
    """Für Training mit Data Augmentation"""
    images = []
    for img in batch["image"]:
        images.append(_transform_train(img))
    batch["image"] = images
    return batch

def _img_to_tensor_val(batch):
    """Für Validation ohne Augmentation"""
    images = []
    for img in batch["image"]:
        images.append(_transform_val(img))
    batch["image"] = images
    return batch

# ÄNDERUNG 3: train_val_split mit optionaler Augmentation ==========
def train_val_split(partition, val_split=0.2, batch_size=32, use_augmentation=True):
    """
    Split partition into train/val and create dataloaders.
    
    Args:
        partition: Dataset partition
        val_split: Validation split ratio
        batch_size: Batch size for dataloaders
        use_augmentation: If True, apply data augmentation to training set
    """
    # Split partition
    num_samples = len(partition)
    num_val_samples = int(num_samples * val_split)
    partition = partition.train_test_split(test_size=num_val_samples, seed=42)
    
    # ÄNDERUNG 4: Unterschiedliche Transforms für Train/Val ==========
    if use_augmentation:
        partition["train"] = partition["train"].with_transform(_img_to_tensor_train)
        print(f"Data Augmentation aktiviert für Training")
    else:
        partition["train"] = partition["train"].with_transform(_img_to_tensor_val)
        print(f"Data Augmentation DEAKTIVIERT für Training")
    
    partition["test"] = partition["test"].with_transform(_img_to_tensor_val)
    
    # Create dataloaders
    train_dataloader = DataLoader(partition["train"], batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(partition["test"], batch_size=batch_size, shuffle=False)
    
    return train_dataloader, val_dataloader

# ÄNDERUNG 5: load_data_split mit Augmentation-Parameter ==========
def load_data_split(partition_id, batch_size, use_augmentation=True):
    """
    Load data for a client partition.
    
    Args:
        partition_id: Client partition ID
        batch_size: Batch size
        use_augmentation: If True, apply data augmentation to training set
    """
    fds = get_fds()
    partition = fds.load_partition(partition_id=partition_id)
    
    return train_val_split(partition, batch_size=batch_size, use_augmentation=use_augmentation)


"""
# Fortgeschrittene Data Augmentation
_transform_train = T.Compose([
    T.RandomRotation(degrees=15),                    # stärkere Rotation
    T.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # stärkere Verschiebung
    T.RandomResizedCrop(size=(28, 28), scale=(0.8, 1.0)),
    T.RandomHorizontalFlip(p=0.3),                   # horizontales Spiegeln (30%)
    T.ColorJitter(brightness=0.2, contrast=0.2),     # Helligkeits-/Kontrastvariation
    T.PILToTensor(),
    T.ConvertImageDtype(torch.float32),
])
"""