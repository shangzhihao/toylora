import torch
import torch.nn as nn

import config
from models import MLP, LoRALinear, LoRAMLP


def test_lora_init_match():
    torch.manual_seed(0)
    in_features, out_features, batch = 4, 3, 5
    base = nn.Linear(in_features, out_features)
    lora = LoRALinear(base, rank=2, alpha=4)

    x = torch.randn(batch, in_features)
    with torch.no_grad():
        y_base = base(x)
        y_lora = lora(x)

    # Because lora_B starts as zeros, adaptation path is zero initially
    assert torch.allclose(y_base, y_lora, atol=1e-6)


def test_lora_requires_grad():
    base = nn.Linear(4, 3)
    lora = LoRALinear(base, rank=2, alpha=4)

    # Original layer is frozen
    assert all(p.requires_grad is False for p in lora.original_layer.parameters())
    # LoRA params are trainable
    assert lora.lora_A.requires_grad is True
    assert lora.lora_B.requires_grad is True


def test_mlp_layers(monkeypatch):
    # Build a base MLP and count its Linear layers
    base = MLP()
    num_linear = sum(isinstance(m, nn.Linear) for m in base.network)

    # When only classifier is adapted
    monkeypatch.setattr(config, "lora_classifer_only", True)
    model_cls_only = LoRAMLP(base, rank=2, alpha=4)
    assert len(model_cls_only.lora_layers) == 1

    # When all linear layers are adapted
    base2 = MLP()
    monkeypatch.setattr(config, "lora_classifer_only", False)
    model_all = LoRAMLP(base2, rank=2, alpha=4)
    assert len(model_all.lora_layers) == num_linear
