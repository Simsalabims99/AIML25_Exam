import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader


def evaluate(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    criterion: torch.nn.Module,
    class_names: list = None,
    flatten: bool = True,
) -> None:
    """
    Evaluates a PyTorch model on a given dataset and prints performance metrics.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        data_loader (DataLoader): DataLoader for the dataset to evaluate (train, val, or test).
        device (torch.device): The device to perform computations (e.g., 'cpu' or 'cuda').
        criterion (torch.nn.Module): The loss function used for evaluation.
        class_names (list): List of class names for the classification report. Defaults to class indices.
        flatten (bool): Whether to flatten input tensors to 1D. Useful for feedforward networks. Default is False.

    Returns:
        None
    """

    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    correct = 0
    total = 0

    all_labels = []  # Collect all true labels
    all_predictions = []  # Collect all predicted labels

    with torch.no_grad():  # Disable gradient computation
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            # Flatten images if necessary
            if flatten:
                images = images.view(images.size(0), -1)

            # Forward pass
            outputs = model(images)

            # Compute loss
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            # Get predictions
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            # Update accuracy metrics
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate metrics
    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = 100 * correct / total

    # Print results
    print("Evaluation Results:")
    print(f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    # Generate and print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))
