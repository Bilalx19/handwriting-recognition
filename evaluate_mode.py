from data import load_data_split
from models.cnn import Net
from train import validate
import torch
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model_performance(partition_id=0, batch_size=32):
    """Evaluiert die Modell-Performance auf einem Client"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Modell initialisieren
    model = Net().to(device)
    
    # Daten laden (OHNE Augmentation für konsistente Evaluation)
    train_loader, val_loader = load_data_split(
        partition_id=partition_id, 
        batch_size=batch_size,
        use_augmentation=False  # Wichtig: Keine Augmentation beim Testen
    )
    
    # Modell evaluieren (ohne vorheriges Training - zufällige Gewichte)
    print("=== Modell mit zufälligen Gewichten ===")
    val_loss, val_accuracy = validate(model, val_loader)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    
    return model, val_loader

def quick_train_and_evaluate(epochs=5, lr=0.001):
    """Kurzes Training + Evaluation für schnellen Test"""
    
    from train import train
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Daten laden
    train_loader, val_loader = load_data_split(
        partition_id=0, 
        batch_size=32,
        use_augmentation=True
    )
    
    # Modell initialisieren
    model = Net().to(device)
    
    print(f"=== Training für {epochs} Epochen ===")
    for epoch in range(epochs):
        train_loss = train(model, train_loader, epochs=1, lr=lr)
        val_loss, val_accuracy = validate(model, val_loader)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    
    return model, val_accuracy

if __name__ == "__main__":
    # Option 1: Nur Evaluation (mit zufälligen Gewichten)
    model, val_loader = evaluate_model_performance(partition_id=0)
    
    # Option 2: Kurzes Training + Evaluation
    print("\n" + "="*50)
    trained_model, final_accuracy = quick_train_and_evaluate(epochs=10, lr=0.001)
    
    print(f"\n🎯 FINALE ACCURACY: {final_accuracy*100:.2f}%")