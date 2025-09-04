import torch
import torch.nn as nn
import torch.optim as optim

import config
from data_util import get_finetune_data_loaders
from pretrain import MLP


class LoRALinear(nn.Module):
    """LoRA adaptation for Linear layers"""

    def __init__(self, original_layer, rank=16, alpha=32):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha

        # Freeze original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False

        # LoRA matrices
        in_features = original_layer.in_features
        out_features = original_layer.out_features

        # A matrix: random initialization
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        # B matrix: zero initialization
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        # Scaling factor
        self.scaling = self.alpha / self.rank

    def forward(self, x):
        # Original output
        original_output = self.original_layer(x)
        # LoRA adaptation: x @ A @ B
        lora_output = (x @ self.lora_A @ self.lora_B) * self.scaling
        return original_output + lora_output


class LoRAMLP(nn.Module):
    """MLP with LoRA adaptations"""

    def __init__(self, pretrained_model, rank=16, alpha=32):
        super().__init__()
        self.pretrained_model = pretrained_model

        # Freeze all pretrained parameters
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        # Replace Linear layers with LoRA versions
        self.lora_layers = nn.ModuleList()

        # Find and replace Linear layers in the sequential network
        for _i, layer in enumerate(self.pretrained_model.network):
            if isinstance(layer, nn.Linear):
                lora_layer = LoRALinear(layer, rank=rank, alpha=alpha)
                self.lora_layers.append(lora_layer)

        # Keep track of layer indices for reconstruction
        self.linear_indices = []
        for i, layer in enumerate(self.pretrained_model.network):
            if isinstance(layer, nn.Linear):
                self.linear_indices.append(i)

    def forward(self, x):
        # Flatten the input (batch_size, 28, 28) -> (batch_size, 784)
        x = x.view(x.size(0), -1)

        lora_idx = 0
        for _i, layer in enumerate(self.pretrained_model.network):
            if isinstance(layer, nn.Linear):
                # Use LoRA adaptation
                x = self.lora_layers[lora_idx](x)
                lora_idx += 1
            else:
                # Use original layer (ReLU, Dropout)
                x = layer(x)

        return x

    def get_lora_parameters(self):
        """Get only LoRA parameters for optimization"""
        lora_params = []
        for lora_layer in self.lora_layers:
            lora_params.extend([lora_layer.lora_A, lora_layer.lora_B])
        return lora_params


def train_lora_model(model, train_loader, criterion, optimizer, device):
    """Train LoRA model for one epoch"""
    model.train()

    for _, (data_, target_) in enumerate(train_loader):
        data, target = data_.to(device), target_.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


def finetune():
    # Set device
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Using device: {device}")

    # Load pretrained model
    pretrained_model = MLP(input_size=784, hidden_sizes=[512, 256, 128], num_classes=10)
    pretrained_model.load_state_dict(
        torch.load(config.pre_model_path, map_location=device)
    )
    pretrained_model.to(device)
    print("Loaded pretrained model from {config.pre_model_path}")

    # Hyperparameters
    lora_rank = config.lora_rank
    lora_alpha = config.lora_alpha
    batch_size = config.lora_batch_size
    learning_rate = config.lora_learning_rate
    num_epochs = config.lora_num_epochs

    # Create LoRA model
    lora_model = LoRAMLP(pretrained_model, rank=lora_rank, alpha=lora_alpha).to(device)
    print(f"Created LoRA model with rank={lora_rank}, alpha={lora_alpha}")

    # Count parameters
    total_params = sum(p.numel() for p in lora_model.parameters())
    trainable_params = sum(
        p.numel() for p in lora_model.parameters() if p.requires_grad
    )
    print(f"Total parameters: {total_params:,}")
    print(
        f"Trainable LoRA parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)"
    )

    # Load data for fine-tuning (digits 8-9)
    print("\nLoading MNIST dataset for fine-tuning...")
    finetune_loader, test_loader = get_finetune_data_loaders(batch_size)

    # Loss function and optimizer (only LoRA parameters)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lora_model.get_lora_parameters(), lr=learning_rate)

    print("\nStarting LoRA fine-tuning...")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_lora_model(lora_model, finetune_loader, criterion, optimizer, device)

    # Save LoRA model
    lora_model_path = config.lora_model_path
    torch.save(lora_model.state_dict(), lora_model_path)
    print(f"\nLoRA model saved as '{lora_model_path}'")

    return lora_model


if __name__ == "__main__":
    finetune()
