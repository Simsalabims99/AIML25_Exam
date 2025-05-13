# Load in relevant libraries, and alias where appropriate
import torch
import torchvision
from torch.utils.data import Dataset, Subset, DataLoader, random_split


def load_torch_data(dataset: str, **kwargs) -> Dataset:
    """Load a dataset from torchvision by its name.

    Args:
        dataset (str): Name of the dataset class to fetch from torchvision.datasets.
        **kwargs: Additional arguments to pass to the dataset constructor.

    Returns:
        Dataset: An instance of the specified torchvision dataset.
    """
    try:
        # Dynamically fetch the dataset class from torchvision.datasets
        dataset_class = getattr(torchvision.datasets, dataset)
        return dataset_class(**kwargs)

    except AttributeError:
        raise ValueError(f"Dataset '{dataset}' not found in torchvision.datasets.")


def to_dataloader(dataset: Dataset, **kwargs) -> DataLoader:
    """Create a dataloader for the dataset"""
    return DataLoader(dataset, **kwargs)


def train_val_split(
    dataset: Dataset, val_ratio: float = 0.2, seed: int = 42
) -> tuple[Subset, Subset]:
    """
    Splits a PyTorch dataset into training and validation datasets.

    Args:
        dataset (torch.utils.data.Dataset): The full dataset to split.
        val_ratio (float): Proportion of the dataset to use for validation (default 0.2).
        seed (int): Random seed for reproducibility (default 42).

    Returns:
        train_dataset (torch.utils.data.Subset): Training subset of the dataset.
        val_dataset (torch.utils.data.Subset): Validation subset of the dataset.
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)

    # Calculate the sizes for training and validation datasets
    total_size = len(dataset)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size

    # Split the dataset into training and validation datasets
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    return train_dataset, val_dataset
