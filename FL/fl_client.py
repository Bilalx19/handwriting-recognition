"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from models.cnn import Net
from train import train as train_fn
from train import validate as test_fn
from data import load_data_split

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    local_epochs=context.run_config["local-epochs"]
    learning_rate = msg.content["config"].get("lr", context.run_config["learning-rate"])
    

    # ==================== ÄNDERUNG 1 ====================
    # NEU: Augmentation aus Run-Config lesen
    use_augmentation = context.run_config.get("use-augmentation", True)
    # NEU: use_augmentation Parameter an load_data_split übergeben
    trainloader, _ = load_data_split(partition_id, batch_size,use_augmentation=use_augmentation)  # <-- NEUER PARAMETER
    # ===================================================
    
    # ==================== ÄNDERUNG 2 ====================
    # NEU: Logging der Augmentation-Einstellung
    print(f"\n Client {partition_id} Konfiguration:")
    print(f"   - Batch Size: {batch_size}")
    print(f"   - Local Epochs: {local_epochs}")
    print(f"   - Learning Rate: {learning_rate}")
    print(f"   - Data Augmentation: {'ON ✅' if use_augmentation else 'OFF ⚠️'}")
    # ===================================================


    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        local_epochs,
        learning_rate,
    )

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
        # ==================== ÄNDERUNG 5 ====================
        # NEU: Metrik ob Augmentation verwendet wurde (für Analyse)
        "augmentation_used": 1 if use_augmentation else 0,
        # ===================================================
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})

    # ==================== ÄNDERUNG 6 ====================
    # NEU: Logging nach Training
    print(f" Client {partition_id} Training abgeschlossen: Loss={train_loss:.4f}")
    # ===================================================

    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]

    
    # ==================== ÄNDERUNG 7 ====================
    # WICHTIG: Validation verwendet NIE Augmentation
    # NEU: Explizit use_augmentation=False gesetzt
    _, valloader = load_data_split(
        partition_id, 
        batch_size,
        use_augmentation=False  # <-- FORCE: Keine Augmentation für Evaluation
    )
    # ===================================================

    # ==================== ÄNDERUNG 8 ====================
    # NEU: Logging der Validierung
    print(f"\n🔍 Client {partition_id} Evaluation:")
    print(f"   - Validation samples: {len(valloader.dataset)}")
    # ===================================================

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})

    # ==================== ÄNDERUNG 9 ====================
    # NEU: Logging der Evaluationsergebnisse
    print(f"Client {partition_id} Evaluation: Loss={eval_loss:.4f}, Acc={eval_acc:.4f}")
    # ===================================================
    return Message(content=content, reply_to=msg)
