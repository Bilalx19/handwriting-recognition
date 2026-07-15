from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

fds = None

# OPTIMIERTE Data Augmentation für FEMNIST (weniger aggressiv)
def get_augmentation_transform():
    """Optimierte Data Augmentation für handschriftliche Zeichen"""
    return T.Compose([
        # Sanfte Rotation (max 5° statt 10°)
        T.RandomRotation(degrees=5, fill=0),
        
        # Geringe Verschiebung (5% statt 10%)
        T.RandomAffine(degrees=0, translate=(0.05, 0.05), fill=0),
        
        # Leichte Skalierung (Zoom)
        T.RandomAffine(degrees=0, scale=(0.95, 1.05)),
        
        # Sehr sanfte Helligkeitsanpassung
        T.ColorJitter(brightness=0.05, contrast=0.05),
        
        # KEINE Perspektive (zu verzerrend)
        # KEIN Mirroring (buchstaben verfälscht)
        
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])

# ALTERNATIVE: Noch konservativere Augmentation
def get_light_augmentation_transform():
    """Sehr leichte Augmentation für empfindliche Zeichen"""
    return T.Compose([
        T.RandomRotation(degrees=3, fill=0),  # Nur 3°
        T.RandomAffine(degrees=0, translate=(0.03, 0.03), fill=0),  # Nur 3%
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])

# ALTERNATIVE: MixUp-artige Augmentation (keine geometrische Änderung)
def get_pixel_augmentation_transform():
    """Nur Pixel-basierte Augmentation ohne geometrische Änderungen"""
    return T.Compose([
        T.ToTensor(),
        T.RandomErasing(p=0.3, scale=(0.02, 0.05), ratio=(0.3, 1.0), value=0),  # Kleine Löcher
        T.Normalize(mean=[0.5], std=[0.5])
    ])

def get_base_transform():
    """Basis-Transformation ohne Augmentation"""
    return T.Compose([
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=[0.5], std=[0.5])
    ])

def get_fds():
    global fds
    if fds is None:
        fds = FederatedDataset(
            dataset="flwrlabs/femnist",
            partitioners={
                "train": IidPartitioner(num_partitions=4)
            },
        )
    return fds

def apply_transform(batch, use_augmentation=False, augmentation_type='standard'):
    images = []
    
    # Wähle Augmentation basierend auf Typ
    if use_augmentation:
        if augmentation_type == 'light':
            transform = get_light_augmentation_transform()
        elif augmentation_type == 'pixel':
            transform = get_pixel_augmentation_transform()
        else:
            transform = get_augmentation_transform()  # standard
    else:
        transform = get_base_transform()
    
    for img in batch["image"]:
        if isinstance(img, torch.Tensor):
            img = T.ToPILImage()(img)
        transformed = transform(img)
        images.append(transformed)
    
    batch["image"] = images
    return batch

def train_val_split(partition, val_split=0.2, batch_size=32, use_augmentation=False, augmentation_type='standard'):
    num_samples = len(partition)
    num_val_samples = int(num_samples * val_split)
    num_train_samples = num_samples - num_val_samples

    partition = partition.train_test_split(test_size=num_val_samples, seed=42)
    
    train_partition = partition["train"].with_transform(
        lambda batch: apply_transform(batch, use_augmentation=use_augmentation, augmentation_type=augmentation_type)
    )
    val_partition = partition["test"].with_transform(
        lambda batch: apply_transform(batch, use_augmentation=False)  # Val immer ohne Augmentation
    )

    train_dataloader = DataLoader(train_partition, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_partition, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader

def load_data_split(partition_id, batch_size, use_augmentation=False, augmentation_type='standard'):
    fds = get_fds()
    partition = fds.load_partition(partition_id=partition_id)
    return train_val_split(partition, batch_size=batch_size, use_augmentation=use_augmentation, augmentation_type=augmentation_type)

#TEST-Funktion für verschiedene Augmentationstypen
def compare_augmentation_types(partition_id=0, num_samples=3):
    """Vergleicht verschiedene Augmentationstypen"""
    fds = get_fds()
    partition = fds.load_partition(partition_id=partition_id)
    
    # Wähle zufällige Samples
    indices = np.random.choice(len(partition), num_samples, replace=False)
    
    fig, axes = plt.subplots(num_samples, 5, figsize=(20, num_samples * 4))
    
    base_transform = get_base_transform()
    aug_standard = get_augmentation_transform()
    aug_light = get_light_augmentation_transform()
    aug_pixel = get_pixel_augmentation_transform()
    
    for i, idx in enumerate(indices):
        sample = partition[idx]
        img = sample["image"]
        char = sample["character"]
        
        if isinstance(img, torch.Tensor):
            img_pil = T.ToPILImage()(img)
        else:
            img_pil = img
        
        # Original
        img_base = base_transform(img_pil)
        axes[i, 0].imshow(img_base.squeeze(), cmap='gray')
        axes[i, 0].set_title(f'Original\nChar: {char}', fontsize=10)
        axes[i, 0].axis('off')
        
        # Standard Augmentation
        img_std = aug_standard(img_pil)
        axes[i, 1].imshow(img_std.squeeze(), cmap='gray')
        axes[i, 1].set_title('Standard\n(Rot±5°, Trans±5%)', fontsize=10)
        axes[i, 1].axis('off')
        
        # Light Augmentation
        img_light = aug_light(img_pil)
        axes[i, 2].imshow(img_light.squeeze(), cmap='gray')
        axes[i, 2].set_title('Light\n(Rot±3°, Trans±3%)', fontsize=10)
        axes[i, 2].axis('off')
        
        # Pixel Augmentation
        img_pixel = aug_pixel(img_pil)
        axes[i, 3].imshow(img_pixel.squeeze(), cmap='gray')
        axes[i, 3].set_title('Pixel\n(Random Erasing)', fontsize=10)
        axes[i, 3].axis('off')
        
        # Keine Augmentation (Kontrolle)
        axes[i, 4].imshow(img_base.squeeze(), cmap='gray')
        axes[i, 4].set_title('Control\n(No Aug)', fontsize=10)
        axes[i, 4].axis('off')
    
    plt.suptitle('Comparison of Augmentation Types for FEMNIST', fontsize=14)
    plt.tight_layout()
    plt.savefig('augmentation_comparison.png', dpi=150, bbox_inches='tight')
    print("Augmentation comparison saved to augmentation_comparison.png")
    plt.show()


def test_augmentation():
    """Testet verschiedene Augmentationstypen"""
    print("\n" + "=" * 60)
    print("TESTING AUGMENTATION TYPES")
    print("=" * 60)
    
    print("\n Comparing augmentation types...")
    compare_augmentation_types(partition_id=0, num_samples=3)
    
    print("\n" + "=" * 60)
    print("EMPFEHLUNGEN:")
    print("=" * 60)

if __name__ == "__main__":
    test_augmentation()
