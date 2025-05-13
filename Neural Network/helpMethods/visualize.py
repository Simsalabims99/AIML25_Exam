import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_training_history(history):
    """
    Plots the training and validation losses and accuracies from the history dictionary.

    Args:
        history (dict): A dictionary containing training history with keys:
                        - 'train_loss': List of training losses.
                        - 'val_loss': List of validation losses.
                        - 'train_acc': List of training accuracies.
                        - 'val_acc': List of validation accuracies.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    # Plot Losses
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)  # Create a subplot for losses
    plt.plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    plt.plot(epochs, history["val_loss"], label="Validation Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    # Plot Accuracies
    plt.subplot(1, 2, 2)  # Create a subplot for accuracies
    plt.plot(epochs, history["train_acc"], label="Train Accuracy", marker="o")
    plt.plot(epochs, history["val_acc"], label="Validation Accuracy", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()
