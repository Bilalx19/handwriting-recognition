"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from models.cnn import Net

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]

    # ==================== ÄNDERUNG 1 ====================
    # NEU: Augmentation-Einstellung aus Run-Config lesen
    use_augmentation: bool = context.run_config.get("use-augmentation", True)
    # ===================================================
    
    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    strategy = FedAvg(fraction_evaluate=fraction_evaluate)

    # Client training configuration
    # ==================== ÄNDERUNG 3 ====================
    # NEU: use_augmentation wird an Clients weitergegeben
    train_config = ConfigRecord({
        "lr": lr,
        "use_augmentation": use_augmentation,  # <-- NEUER EINTRAG
    })

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=train_config,
        num_rounds=num_rounds,
    )
    

    # Save final model to disk
    print("\nSaving final model to disk...")

    #IF ELSE VERSCHÖNERT, keine LOGIK geändert
    final_arrays = result.arrays if result.arrays is not None else arrays
    state_dict = final_arrays.to_torch_state_dict()

    print("Saved keys:", state_dict.keys())

    # ==================== ÄNDERUNG 6 ====================
    # NEU: Dynamischer Dateiname mit Augmentation-Info
    aug_suffix = "with_aug" if use_augmentation else "no_aug"
    model_filename = f"final_model_{aug_suffix}_rounds_{num_rounds}.pt"
    # ===================================================

    torch.save(state_dict, "final_model.pt")
