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

        self._freeze_backbone()
        self.linear_indices = self._collect_linear_indices()
        self.apply_linear_indices = self._select_target_layers()
        self.lora_layers = self._build_lora_layers(rank, alpha)

    def _freeze_backbone(self):
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

    def _collect_linear_indices(self):
        return [
            i
            for i, layer in enumerate(self.pretrained_model.network)
            if isinstance(layer, nn.Linear)
        ]

    def _select_target_layers(self):
        if config.lora_classifer_only and self.linear_indices:
            return {self.linear_indices[-1]}
        return set(self.linear_indices)

    def _build_lora_layers(self, rank, alpha):
        layers = []
        for i, layer in enumerate(self.pretrained_model.network):
            if isinstance(layer, nn.Linear) and i in self.apply_linear_indices:
                layers.append(LoRALinear(layer, rank=rank, alpha=alpha))
        return nn.ModuleList(layers)

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
