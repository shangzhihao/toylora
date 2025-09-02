import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset


class MLP(nn.Module):
    """Multi-Layer Perceptron for MNIST classification"""
    
    def __init__(self, input_size=784, hidden_sizes=[512, 256, 128], num_classes=10, dropout_rate=0.2):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        # Create hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        # Flatten the input (batch_size, 28, 28) -> (batch_size, 784)
        x = x.view(x.size(0), -1)
        return self.network(x)


def filter_dataset_by_labels(dataset, target_labels):
    """Filter dataset to only include samples with specified labels"""
    indices = []
    for i, (_, label) in enumerate(dataset):
        if label in target_labels:
            indices.append(i)
    return Subset(dataset, indices)


def load_mnist_data(batch_size=64, train_labels=None):
    """Load and preprocess MNIST dataset
    
    Args:
        batch_size: Batch size for data loaders
        train_labels: List of labels to include in training set. If None, include all labels.
    """
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Download and load training data
    full_train_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # Download and load test data (always use full test set for evaluation)
    test_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Filter training dataset if train_labels is specified
    if train_labels is not None:
        train_dataset = filter_dataset_by_labels(full_train_dataset, train_labels)
        print(f"Filtered training set to labels {train_labels}")
        print(f"Training samples: {len(train_dataset)} (filtered from {len(full_train_dataset)})")
    else:
        train_dataset = full_train_dataset
        print(f"Training samples: {len(train_dataset)} (all labels)")
    
    print(f"Test samples: {len(test_dataset)} (all labels for evaluation)")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def train_model(model, train_loader, criterion, optimizer, device):
    """Train the model for one epoch"""
    model.train()
    
    for _, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


def evaluate_model_by_class(model, test_loader, criterion, device):
    """Evaluate the model on test data with per-class breakdown"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    # Track per-class accuracy
    class_correct = [0] * 10
    class_total = [0] * 10
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            test_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Per-class accuracy tracking
            c = (predicted == target).squeeze()
            for i in range(target.size(0)):
                label = target[i]
                class_correct[label] += c[i].item() if c.dim() > 0 else c.item()
                class_total[label] += 1
    
    test_loss /= len(test_loader)
    test_acc = 100. * correct / total
    
    # Calculate per-class accuracies
    class_accuracies = {}
    for i in range(10):
        if class_total[i] > 0:
            class_accuracies[i] = 100. * class_correct[i] / class_total[i]
        else:
            class_accuracies[i] = 0.0
    
    return test_loss, test_acc, class_accuracies



def pretrain():
    # Set device
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    print(f'Using device: {device}')
    
    # Hyperparameters
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 10
    hidden_sizes = [512, 256, 128]
    dropout_rate = 0.2
    
    # Define training labels (0-7 only)
    train_labels = list(range(8))  # [0, 1, 2, 3, 4, 5, 6, 7]
    print(f"Training only on digits: {train_labels}")
    print("Evaluating on all digits (0-9) including unseen digits 8 and 9")
    
    # Load data
    print("\nLoading MNIST dataset...")
    train_loader, test_loader = load_mnist_data(batch_size, train_labels=train_labels)
    
    # Create model
    model = MLP(
        input_size=784,
        hidden_sizes=hidden_sizes,
        # Still 10 classes for output
        # even though we only train on 8
        num_classes=10,
        dropout_rate=dropout_rate
    ).to(device)
    
    print(f"\nModel architecture: {model}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_model(model, train_loader, criterion, optimizer, device)
    
    # Save the model
    model_path = 'mnist_mlp_pretrained.pth'
    torch.save(model.state_dict(), model_path)
    print(f"\nPretrained model saved as '{model_path}'")
    
    return model
if __name__ == "__main__":
    pretrain()