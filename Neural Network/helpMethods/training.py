import torch
from torch.utils.data import DataLoader
from torch._prims_common import DeviceLikeType


def fit(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: DeviceLikeType,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    num_epochs: int,
    flatten: bool = True,
) -> tuple[torch.nn.Module, dict[str, list]]:
    """
    Trains and validates a PyTorch neural network model.

    Args:
        model (torch.nn.Module): The neural network model to be trained.
        train_loader (DataLoader): DataLoader for the training dataset. It provides batches of data.
        val_loader (DataLoader): DataLoader for the validation dataset. It provides batches of data.
        device (DeviceLikeType): The device to run the training (e.g., 'cpu' or 'cuda').
        optimizer (torch.optim.Optimizer): Optimizer to update the model's parameters based on the gradients.
        criterion (torch.nn.Module): The loss function to compute the error between predictions and true labels.
        num_epochs (int): Number of epochs (full passes through the training data) to train the model.
        flatten (bool): Whether to flatten input tensors to 1D. Useful for feedforward networks. Default is False.

    Returns:
        Dict[str, list]: A dictionary containing training and validation loss and accuracy histories:
            - 'train_loss': List of training losses for each epoch.
            - 'val_loss': List of validation losses for each epoch.
            - 'train_acc': List of training accuracies for each epoch.
            - 'val_acc': List of validation accuracies for each epoch.

    Notes:
        This function alternates between training and validation phases:
        1. During the training phase, the model is updated using backpropagation and gradient descent.
        2. During the validation phase, the model's performance is evaluated without updating parameters.
        Metrics for each phase are printed for every epoch.

    Example Usage:
        train_history = fit(
            model=my_model,
            train_loader=my_train_loader,
            val_loader=my_val_loader,
            device='cuda',
            optimizer=my_optimizer,
            criterion=torch.nn.CrossEntropyLoss(),
            num_epochs=10
        )
    """
    # Initialize history trackers
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(num_epochs):
        #################
        # TRAINING PHASE
        #################

        model.train()  # Set the model to training mode

        train_loss = 0.0  # Accumulate the training loss
        correct = 0  # Count of correctly predicted samples
        total = 0  # Total number of samples

        for (
            images,
            labels,
        ) in train_loader:  # Loop through batches in the training dataset
            # Move data to the selected device (CPU or GPU)
            images, labels = images.to(device), labels.to(device)

            # Flatten the images into 1D vectors (if necessary for fully connected layers)
            if flatten:
                images = images.view(images.size(0), -1)

            # Forward pass: Compute model outputs and loss
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass: Compute gradients
            optimizer.zero_grad()  # Clear previous gradients
            loss.backward()  # Compute new gradients

            # Update model parameters
            optimizer.step()

            # Accumulate loss and accuracy metrics
            train_loss += loss.item() * images.size(0)  # Loss multiplied by batch size
            _, predicted = torch.max(outputs, 1)  # Get predicted class
            total += labels.size(0)  # Update total number of samples
            correct += (predicted == labels).sum().item()  # Count correct predictions

        # Calculate average training loss and accuracy
        train_loss /= len(train_loader.dataset)
        train_accuracy = 100 * correct / total

        ###################
        # VALIDATION PHASE
        ###################

        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0  # Accumulate validation loss
        correct = 0
        total = 0

        with torch.no_grad():  # No gradients needed during validation
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                if flatten:
                    images = images.view(images.size(0), -1)

                outputs = model(images)  # Forward pass
                loss = criterion(outputs, labels)  # Compute loss

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Calculate average validation loss and accuracy
        val_loss /= len(val_loader.dataset)
        val_accuracy = 100 * correct / total

        # Save metrics to history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_accuracy)
        history["val_acc"].append(val_accuracy)

        # Print progress for the current epoch
        print(
            f"Epoch [{epoch+1}/{num_epochs}]: "
            f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%"
        )

    return model, history
