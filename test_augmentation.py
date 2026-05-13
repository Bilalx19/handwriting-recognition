from data import load_data_split
import matplotlib.pyplot as plt
import torch

# Lade Daten mit Augmentation
train_loader_aug, val_loader = load_data_split(partition_id=0, batch_size=32, use_augmentation=True)

# Zeige ein Batch
batch = next(iter(train_loader_aug))
images = batch["image"]
labels = batch["character"]

# Visualisiere augmentierte Bilder
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    if i < len(images):
        img = images[i].squeeze().numpy()
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Label: {labels[i].item()}")
        ax.axis('off')
plt.suptitle("Augmentierte Trainingsbilder")
plt.show()

# Lade Daten OHNE Augmentation zum Vergleich
train_loader_noaug, _ = load_data_split(partition_id=0, batch_size=32, use_augmentation=False)
batch_noaug = next(iter(train_loader_noaug))
images_noaug = batch_noaug["image"]

# Zeige originale Bilder
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    if i < len(images_noaug):
        img = images_noaug[i].squeeze().numpy()
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Original")
        ax.axis('off')
plt.suptitle("Originale (nicht augmentierte) Bilder")
plt.show()