import torch

import config
from data_util import get_inference_data_loader
from models import MLP, LoRAMLP


def evaluate_model_accuracy(model, test_loader, device):
    """Evaluate model and return per-class accuracies"""
    model.eval()

    # Track per-class accuracy
    class_correct = [0] * 10
    class_total = [0] * 10
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for data_, target_ in test_loader:
            data, target = data_.to(device), target_.to(device)

            output = model(data)
            _, predicted = torch.max(output, 1)

            total_samples += target.size(0)
            total_correct += (predicted == target).sum().item()

            # Per-class accuracy tracking
            c = (predicted == target).squeeze()
            for i in range(target.size(0)):
                label = target[i]
                class_correct[label] += c[i].item() if c.dim() > 0 else c.item()
                class_total[label] += 1

    # Calculate accuracies
    overall_accuracy = 100.0 * total_correct / total_samples
    class_accuracies = {}

    for i in range(10):
        if class_total[i] > 0:
            class_accuracies[i] = 100.0 * class_correct[i] / class_total[i]
        else:
            class_accuracies[i] = 0.0

    return overall_accuracy, class_accuracies


def print_accuracy_results(model_name, overall_accuracy, class_accuracies):
    """Print formatted accuracy results"""
    print(f"\n{model_name} Results:")
    print("=" * 50)
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")
    print("\nPer-class Accuracies:")
    for digit in range(10):
        print(f"  Digit {digit}: {class_accuracies[digit]:.2f}%")


def inference():
    # Set device
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Using device: {device}")

    # 1. Load dataset
    print("\nLoading MNIST test dataset...")
    test_loader = get_inference_data_loader(batch_size=64)

    # 2. Load pretrained model
    print("\nLoading pretrained model...")
    pretrained_model = MLP(input_size=784, hidden_sizes=[512, 256, 128], num_classes=10)
    pretrained_model.load_state_dict(
        torch.load(config.pre_model_path, map_location=device)
    )
    pretrained_model.to(device)
    print(f"✓ Successfully loaded {config.pre_model_path}")

    # 3. Evaluate pretrained model
    print("\nEvaluating pretrained model...")
    pretrained_accuracy, pretrained_class_acc = evaluate_model_accuracy(
        pretrained_model, test_loader, device
    )
    print_accuracy_results(
        "Pretrained Model", pretrained_accuracy, pretrained_class_acc
    )

    # 4. Load LoRA fine-tuned model
    print("\nLoading LoRA fine-tuned model...")
    # Create LoRA model structure
    lora_model = LoRAMLP(
        pretrained_model, rank=config.lora_rank, alpha=config.lora_alpha
    )
    lora_model.load_state_dict(torch.load(config.lora_model_path, map_location=device))
    lora_model.to(device)
    print(f"✓ Successfully loaded {config.lora_model_path}")

    # 5. Evaluate LoRA fine-tuned model
    print("\nEvaluating LoRA fine-tuned model...")
    lora_accuracy, lora_class_acc = evaluate_model_accuracy(
        lora_model, test_loader, device
    )
    print_accuracy_results("LoRA Fine-tuned Model", lora_accuracy, lora_class_acc)

    # Comparison summary
    print("\nComparison Summary:")
    print("=" * 70)
    print(f"{'Digit':<8} {'Pretrained':<12} {'LoRA Finetuned':<15} {'Improvement':<12}")
    print("-" * 70)

    for digit in range(10):
        pretrained_acc = pretrained_class_acc[digit]
        lora_acc = lora_class_acc[digit]
        improvement = lora_acc - pretrained_acc

        print(
            f"{digit:<8} {pretrained_acc:<12.2f} {lora_acc:<15.2f} {improvement:<12.2f}"
        )

    overall_improvement = lora_accuracy - pretrained_accuracy
    print("-" * 70)
    print(
        f"{'Overall':<8} {pretrained_accuracy:<12.2f} {lora_accuracy:<15.2f} {overall_improvement:<12.2f}"
    )
    return pretrained_accuracy, lora_accuracy


if __name__ == "__main__":
    inference()
