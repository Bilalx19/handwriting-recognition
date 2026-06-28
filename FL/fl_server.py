import json
import torch
from datetime import datetime, timezone
from pathlib import Path

from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from models.cnn import Net


METRICS_FILE = Path("/app/metrics/server_metrics.json")
MODEL_FILE = Path("/app/model/final_model.pt")


def _append_metric(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if path.exists():
        with open(path) as f:
            existing = json.load(f)
    existing.append(record)
    with open(path, "w") as f:
        json.dump(existing, f)


class TrackingFedAvg(FedAvg):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._fit_contributions: dict = {}

    def aggregate_evaluate(self, server_round, results):
        results = list(results)
        ret = super().aggregate_evaluate(server_round, results)

        total = sum(int(msg.content["metrics"]["num-examples"]) for msg in results)
        if total > 0:
            eval_loss = sum(
                float(msg.content["metrics"]["eval_loss"]) * int(msg.content["metrics"]["num-examples"])
                for msg in results
            ) / total
            eval_acc = sum(
                float(msg.content["metrics"]["eval_acc"]) * int(msg.content["metrics"]["num-examples"])
                for msg in results
            ) / total
            _append_metric(METRICS_FILE, {
                "round": server_round,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "global_eval_loss": eval_loss,
                "global_eval_acc": eval_acc,
                "num_clients": len(results),
                "client_contributions": self._fit_contributions,
            })

        return ret


app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]

    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    strategy = TrackingFedAvg(fraction_evaluate=fraction_evaluate)

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
    )

    final_arrays = result.arrays if result.arrays is not None else arrays
    state_dict = final_arrays.to_torch_state_dict()

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, str(MODEL_FILE))
    print(f"Model saved to {MODEL_FILE}")
