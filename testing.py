from data import load_data_split


train_dataloader, val_dataloader = load_data_split(partition_id=0, batch_size=32)

batch = next(iter(train_dataloader))

images = batch["image"]
true_char = batch["character"]

print(images.shape)
print(true_char.shape)
print(true_char[:10])
print(f"train batches: {len(train_dataloader)}")
print(f"validation batches: {len(val_dataloader)}")
