import json
import torch
from datetime import datetime, timezone
from pathlib import Path

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from models.cnn import Net
from train import train as train_fn
from train import validate as test_fn
from data import load_data_split


METRICS_DIR = Path("/app/metrics")

app = ClientApp()


def _append_client_metric(partition_id: int, record: dict) -> None:
    path = METRICS_DIR / f"client_{partition_id}_metrics.json"
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    existing = []
    if path.exists():
        with open(path) as f:
            existing = json.load(f)
    existing.append(record)
    with open(path, "w") as f:
        json.dump(existing, f)


@app.train()
def train(msg: Message, context: Context):
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    trainloader, _ = load_data_split(partition_id, batch_size)

    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"].get("lr", context.run_config["learning-rate"]),
    )

    _append_client_metric(int(partition_id), {
        "type": "train",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "train_loss": float(train_loss),
        "num_examples": len(trainloader.dataset),
    })

    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    _, valloader = load_data_split(partition_id, batch_size)

    eval_loss, eval_acc = test_fn(model, valloader)

    _append_client_metric(int(partition_id), {
        "type": "eval",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "eval_loss": float(eval_loss),
        "eval_acc": float(eval_acc),
        "num_examples": len(valloader.dataset),
    })

    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
