# ToyLoRA: Understanding Low-Rank Adaptation through MNIST

A toy implementation demonstrating how **LoRA (Low-Rank Adaptation)** works. This project shows the core concepts of LoRA by:

1. **Base Training**: Training an MLP on MNIST digits 0-7
2. **LoRA Fine-tuning**: Using low-rank adaptation to efficiently learn digits 8-9 without modifying the original model weights

## 🎯 What is LoRA?

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that:
- Keeps the original pre-trained model weights **frozen**
- Adds small, trainable low-rank matrices to learn new tasks
- Achieves comparable performance with far fewer trainable parameters
- Allows multiple task-specific adaptations without model duplication

## 🧠 Demonstration

This toy example illustrates LoRA concepts through a simple scenario:

### Phase 1: Base Model Training
- **Dataset**: MNIST digits 0-7 only (48,200 samples)
- **Architecture**: Standard MLP (784 → 512 → 256 → 128 → 10)
- **Result**: Model learns to classify digits 0-7 with ~98.7% accuracy
- **Problem**: Cannot classify digits 8-9 (0% accuracy on unseen classes)

### Phase 2: LoRA Fine-tuning
- **New Task**: Learn to classify digits 8-9
- **LoRA Approach**: Add low-rank matrices A and B to existing layers
- **Key Insight**: Original weights stay frozen, only LoRA parameters train
- **Efficiency**: Much fewer parameters than full fine-tuning

## 🏗️ Overview

```
Base MLP (frozen after Phase 1):
Input(784) → Linear₁(512) → ReLU → Dropout → 
             Linear₂(256) → ReLU → Dropout → 
             Linear₃(128) → ReLU → Dropout → 
             Output(10)

LoRA Adaptation (Phase 2):
For each Linear layer:
  Original: W × x
  With LoRA: (W + A × B) × x
  Where: A ∈ ℝᵐˣʳ, B ∈ ℝʳˣⁿ, r << min(m,n)
```

Lora is a quit easy theory. `W` is the parameter matrix of pretrained model and trained in the pretrain phase. In the finetune phase, we create a new model with parameters `W'=(W + A × B)`. `A` is initialized randomly and `B` is initialized as zeors, so `W'=W` when lora model is created. `A` and `B` are trainable matrices and `W` is frozen in the finetune phase. There are `mˣn` parameters to be finetuned without lora, and `mˣr+rˣn` parameters to be finetuned with lora. The total number of parameters is reduced, and the training time is reduced if `r` is small.

## 🚀 Quick Start

### Installation
```bash
uv sync
```
### Pretrain Phase: Train Base Model (digits 0-7)
```bash
python main.py --job=pretrain
or
python ptrain.py
```

This creates:
- `pretrained.pth` - 567,434 trainable parameters
- Training on digits 0-7 only

### Finetune Phase: LoRA Fine-tuning (digits 8-9)
```bash
python main.py --job=finetune
or
python finetune.py
```
This creates:
- `finetuned.pth` - 567,434 frozen parameters and 41,367 trainable parameters
- Training on digits 8 and 9

### Testing and Inference
```bash
python main.py --job=inference
or
python inference.py
```

## 📊 Expected Results

### Base Model Performance
```
Trained Classes (0-7): 98.74% average accuracy
Unseen Classes (8-9):   0.00% accuracy
Overall Accuracy:      79.17%
```

### After LoRA Fine-tuning
```
Original Classes (0-7): ~98.7% (maintained)
New Classes (8-9):      ~95%+ (learned via LoRA)
Overall Accuracy:       ~98%+
LoRA Parameters:        6.8% of total model parameters
```

> ⚠️ **WARNING**: Catastrophic forgetting is currently occurring in this implementation. The model may not achieve the expected results shown above. This issue is being actively resolved.


## 📚 Further Reading

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Parameter-Efficient Fine-Tuning Methods](https://arxiv.org/abs/2110.04366)
- [Understanding Low-Rank Adaptations](https://arxiv.org/abs/2106.09685)

## 📄 License

This project is licensed under the terms specified in the LICENSE file.