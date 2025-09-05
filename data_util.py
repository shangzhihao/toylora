"""
Data utilities for MNIST dataset operations in the LoRA project.

This module contains all dataset-related operations including loading,
preprocessing, filtering, and creating data loaders for the MNIST dataset.
"""

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

import config


def get_mnist_transforms():
    """Get standard MNIST data transforms for normalization"""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean and std
        ]
    )


def filter_dataset_by_labels(dataset, target_labels):
    """Filter dataset to only include samples with specified labels

    Args:
        dataset: PyTorch dataset to filter
        target_labels: List of labels to include in the filtered dataset

    Returns:
        Subset: Filtered dataset containing only specified labels
    """
    indices = []
    for i, (_, label) in enumerate(dataset):
        if label in target_labels:
            indices.append(i)
    return Subset(dataset, indices)


def load_mnist_datasets(train_labels=None):
    """Load MNIST training and test datasets with optional filtering

    Args:
        train_labels: List of labels to include in training set. If None, include all labels.

    Returns:
        tuple: (train_dataset, test_dataset)
    """
    transform = get_mnist_transforms()

    # Download and load training data
    full_train_dataset = torchvision.datasets.MNIST(
        root=config.data_path, train=True, download=True, transform=transform
    )

    # Download and load test data (always use full test set for evaluation)
    test_dataset = torchvision.datasets.MNIST(
        root=config.data_path, train=False, download=True, transform=transform
    )

    # Filter training dataset if train_labels is specified
    if train_labels is not None:
        train_dataset = filter_dataset_by_labels(full_train_dataset, train_labels)
        print(f"Filtered training set to labels {train_labels}")
        print(
            f"Training samples: {len(train_dataset)} (filtered from {len(full_train_dataset)})"
        )
    else:
        train_dataset = full_train_dataset
        print(f"Training samples: {len(train_dataset)} (all labels)")

    print(f"Test samples: {len(test_dataset)} (all labels for evaluation)")

    return train_dataset, test_dataset


def create_data_loaders(train_dataset, test_dataset, batch_size=64):
    """Create data loaders for training and testing datasets

    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset
        batch_size: Batch size for data loaders

    Returns:
        tuple: (train_loader, test_loader)
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def load_mnist_data(batch_size=64, train_labels=None):
    """Load and preprocess MNIST dataset with data loaders

    Args:
        batch_size: Batch size for data loaders
        train_labels: List of labels to include in training set. If None, include all labels.

    Returns:
        tuple: (train_loader, test_loader)
    """
    train_dataset, test_dataset = load_mnist_datasets(train_labels)
    return create_data_loaders(train_dataset, test_dataset, batch_size)


def load_test_dataset(batch_size=64):
    """Load MNIST test dataset for inference

    Args:
        batch_size: Batch size for the test data loader

    Returns:
        DataLoader: Test data loader
    """
    transform = get_mnist_transforms()

    test_dataset = torchvision.datasets.MNIST(
        root=config.data_path, train=False, download=True, transform=transform
    )

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"Test samples: {len(test_dataset)}")
    return test_loader


def get_pretrain_data_loaders(batch_size=None):
    """Get data loaders for pretraining phase (digits 0-7 only)

    Args:
        batch_size: Batch size for data loaders. If None, uses config default.

    Returns:
        tuple: (train_loader, test_loader)
    """
    if batch_size is None:
        batch_size = config.pre_batch_size

    train_labels = list(range(8))  # [0, 1, 2, 3, 4, 5, 6, 7]
    print(f"Training only on digits: {train_labels}")
    print("Evaluating on all digits (0-9) including unseen digits 8 and 9")

    return load_mnist_data(batch_size, train_labels=train_labels)


def get_finetune_data_loaders(batch_size=None):
    """Get data loaders for fine-tuning phase (digits 8-9 only)

    Args:
        batch_size: Batch size for data loaders. If None, uses config default.

    Returns:
        tuple: (train_loader, test_loader)
    """
    if batch_size is None:
        batch_size = config.lora_batch_size

    finetune_labels = [8, 9]
    print(f"Fine-tuning on digits: {finetune_labels}")

    return load_mnist_data(batch_size, train_labels=finetune_labels)


def get_inference_data_loader(batch_size=64):
    """Get data loader for inference phase (all digits)

    Args:
        batch_size: Batch size for the data loader

    Returns:
        DataLoader: Test data loader for inference
    """
    return load_test_dataset(batch_size)
