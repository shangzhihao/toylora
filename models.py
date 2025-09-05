"""
Neural network models for the LoRA demonstration project.

This module contains all PyTorch model classes used in the educational
LoRA implementation, including the base MLP and LoRA-adapted versions.
"""

import torch
import torch.nn as nn
import config


class MLP(nn.Module):
    """Multi-Layer Perceptron for MNIST classification"""

    def __init__(
        self,
        input_size=784,
        hidden_sizes=None,
        num_classes=10,
        dropout_rate=0.2,
    ):
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [512, 256, 128]

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

        # Identify indices of Linear layers in the backbone
        linear_indices = []
        for i, layer in enumerate(self.pretrained_model.network):
            if isinstance(layer, nn.Linear):
                linear_indices.append(i)

        # Decide which Linear layers to adapt with LoRA
        if config.lora_classifer_only and len(linear_indices) > 0:
            self.apply_linear_indices = {linear_indices[-1]}
        else:
            self.apply_linear_indices = set(linear_indices)

        # Create LoRA modules only for selected Linear layers
        self.lora_layers = nn.ModuleList()
        for i, layer in enumerate(self.pretrained_model.network):
            if isinstance(layer, nn.Linear) and i in self.apply_linear_indices:
                lora_layer = LoRALinear(layer, rank=rank, alpha=alpha)
                self.lora_layers.append(lora_layer)

        # Keep track of all Linear layer indices (for reference)
        self.linear_indices = linear_indices

    def forward(self, x):
        # Flatten the input (batch_size, 28, 28) -> (batch_size, 784)
        x = x.view(x.size(0), -1)

        lora_idx = 0
        for i, layer in enumerate(self.pretrained_model.network):
            if isinstance(layer, nn.Linear):
                if i in self.apply_linear_indices:
                    # Use LoRA adaptation for selected layers
                    x = self.lora_layers[lora_idx](x)
                    lora_idx += 1
                else:
                    # Pass through the original frozen Linear layer
                    x = layer(x)
            else:
                # Use original non-Linear layers (ReLU, Dropout)
                x = layer(x)

        return x

    def get_lora_parameters(self):
        """Get only LoRA parameters for optimization"""
        lora_params = []
        for lora_layer in self.lora_layers:
            lora_params.extend([lora_layer.lora_A, lora_layer.lora_B])
        return lora_params
