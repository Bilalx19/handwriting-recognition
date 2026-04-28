from data import load_data

dataloader = load_data(partition_id=0, batch_size=32)

batch = next(iter(dataloader))

images = batch["image"]
true_char = batch["character"]

print(images.shape)
print(true_char.shape)
print(true_char[:10])