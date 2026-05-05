import torch

from data import load_data_split
from models.cnn import Net

# device is set at module level, all functions use this
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {'GPU' if device.type == 'cuda' else 'CPU'} for training")


def train(net, train_dataloader, epochs, lr):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    # SGD used over Adam, because Federated Averaging averages model weights across clients.
    # Adam maintains internal state (momentum, variance) that is not averaged across clients,
    # which can lead to divergence in training. SGD has no such state and is fully compatible
    # with federated averaging.
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in train_dataloader:
            images = batch["image"].to(device)
            labels = batch["character"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_train_loss = running_loss / (epochs * len(train_dataloader))
    return avg_train_loss


def validate(net, val_dataloader):
    """Evaluate the model on the validation set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0

    net.eval()
    with torch.no_grad():
        for batch in val_dataloader:
            images = batch["image"].to(device)
            labels = batch["character"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs, 1)[1] == labels).sum().item()
    net.train()

    accuracy = correct / len(val_dataloader.dataset)
    loss = loss / len(val_dataloader)
    return loss, accuracy


if __name__ == "__main__":
    # dataset_partition = load_from_disk("./datasets/partition_0")
    # print(dataset_partition) 

    # Sanity check: local training for partition 0.
    PARTITION_ID = 0
    EPOCHS = 10
    LR = 0.01
    BATCH_SIZE = 32

    train_loader, val_loader = load_data_split(PARTITION_ID, BATCH_SIZE)

    net = Net().to(device)

    for epoch in range(EPOCHS):
        train_loss = train(net, train_loader, epochs=1, lr=LR)
        val_loss, val_accuracy = validate(net, val_loader)
        print(f"Epoch {epoch + 1}/{EPOCHS} — train loss: {train_loss:.4f} | val loss: {val_loss:.4f} | val accuracy: {val_accuracy:.4f}")