import json
import torch
import torch.nn.functional as F
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


def _flatten(state_dict: dict) -> torch.Tensor:
    return torch.cat([v.float().flatten() for v in state_dict.values()])


class TrackingFedAvg(FedAvg):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._fit_contributions: dict = {}
        self._global_weights: dict | None = None

    def aggregate_train(self, server_round, results):
        results = list(results)

        if self._global_weights is not None and results:
            global_flat = _flatten(self._global_weights)

            client_states = {}
            client_samples = {}
            for msg in results:
                node_id = str(msg.metadata.src_node_id)
                n = int(msg.content["metrics"]["num-examples"])
                client_states[node_id] = msg.content["arrays"].to_torch_state_dict()
                client_samples[node_id] = n

            total = sum(client_samples.values())

            # fedavg aggregation to get new global weights
            first = next(iter(client_states.values()))
            new_global = {
                key: sum(
                    client_states[nid][key].float() * client_samples[nid]
                    for nid in client_states
                ) / total
                for key in first
            }

            agg_delta = _flatten(new_global) - global_flat

            contributions = {}
            for node_id, state in client_states.items():
                delta = _flatten(state) - global_flat
                norm = delta.norm().item()
                if norm > 0 and agg_delta.norm().item() > 0:
                    cos_sim = F.cosine_similarity(
                        delta.unsqueeze(0), agg_delta.unsqueeze(0)
                    ).item()
                else:
                    cos_sim = 0.0
                contributions[node_id] = {
                    "l2_norm": round(norm, 4),
                    "cosine_sim": round(cos_sim, 4),
                }

            self._fit_contributions = contributions
            self._global_weights = new_global

        return super().aggregate_train(server_round, results)

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
    strategy._global_weights = {k: v.clone() for k, v in global_model.state_dict().items()}

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
