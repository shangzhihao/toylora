import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, Subset

import config
from data_util import get_finetune_data_loaders, load_mnist_datasets
from models import MLP, LoRAMLP


def train_lora_model(model, train_loader, criterion, optimizer, device):
    """Train LoRA model for one epoch."""
    model.train()

    for _, (data_, target_) in enumerate(train_loader):
        data, target = data_.to(device), target_.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


def resolve_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_pretrained_model(device):
    model = MLP(input_size=784, hidden_sizes=[512, 256, 128], num_classes=10)
    state_dict = torch.load(config.pre_model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    return model


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def maybe_mix_old_digits(loader, batch_size):
    if not getattr(config, "lora_mix_old_digits", False):
        return loader

    print("\nMixing digits [0-7] into the fine-tuning set...")
    train_89, _ = load_mnist_datasets(train_labels=[8, 9])
    train_07, _ = load_mnist_datasets(train_labels=list(range(8)))

    num_89 = len(train_89)
    num_mix = max(1, int(config.lora_digits_per * num_89))
    num_mix = min(num_mix, len(train_07))

    sampled_indices = random.sample(range(len(train_07)), num_mix)
    subset_07 = Subset(train_07, sampled_indices)

    combined_train = ConcatDataset([train_89, subset_07])
    print(
        f"Fine-tuning samples [8,9]: {num_89}, mixed [0-7]: {len(subset_07)}"
    )
    print(f"Total fine-tuning samples: {len(combined_train)}")
    return DataLoader(combined_train, batch_size=batch_size, shuffle=True)


def finetune():
    device = resolve_device()
    print(f"Using device: {device}")

    pretrained_model = load_pretrained_model(device)
    print(f"Loaded pretrained model from {config.pre_model_path}")

    lora_rank = config.lora_rank
    lora_alpha = config.lora_alpha
    batch_size = config.lora_batch_size
    learning_rate = config.lora_learning_rate
    num_epochs = config.lora_num_epochs

    lora_model = LoRAMLP(pretrained_model, rank=lora_rank, alpha=lora_alpha).to(device)
    print(f"Created LoRA model with rank={lora_rank}, alpha={lora_alpha}")

    total_params, trainable_params = count_parameters(lora_model)
    print(f"Total parameters: {total_params:,}")
    print(
        f"Trainable LoRA parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)"
    )

    print("\nLoading MNIST dataset for fine-tuning...")
    finetune_loader, _ = get_finetune_data_loaders(batch_size)
    finetune_loader = maybe_mix_old_digits(finetune_loader, batch_size)

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
