import torch
import torch.nn as nn
import torch.optim as optim

import config
from data_util import get_pretrain_data_loaders
from models import MLP


def train_model(model, train_loader, criterion, optimizer, device):
    """Train the model for one epoch"""
    model.train()

    for _, (data_, target_) in enumerate(train_loader):
        data, target = data_.to(device), target_.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


def pretrain():
    # Set device
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = config.pre_batch_size
    learning_rate = config.pre_learning_rate
    num_epochs = config.pre_num_epochs
    hidden_sizes = config.pre_hidden_sizes
    dropout_rate = config.pre_dropout_rate
    model_path = config.pre_model_path

    # Load data
    print("\nLoading MNIST dataset...")
    train_loader, test_loader = get_pretrain_data_loaders(batch_size)

    # Create model
    model = MLP(
        # MINST is 28x28
        input_size=784,
        hidden_sizes=hidden_sizes,
        # Still 10 classes for output
        # even though we only train on 8
        num_classes=10,
        dropout_rate=dropout_rate,
    ).to(device)

    print(f"\nModel architecture: {model}")

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    print("\nStarting training...")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_model(model, train_loader, criterion, optimizer, device)

    # Save the model
    torch.save(model.state_dict(), model_path)
    print(f"\nPretrained model saved as '{model_path}'")

    return model


if __name__ == "__main__":
    pretrain()
