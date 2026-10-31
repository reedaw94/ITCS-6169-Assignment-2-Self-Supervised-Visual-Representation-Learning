# ITCS-6169-Assignment-2-Self-Supervised-Visual-Representation-Learning
ResNet-18 scene classification using supervised and SimCLR self-supervised learning, with contrastive pretraining, fine-tuning, and performance evaluation across 16 scene categories.

# ITCS 6169 Assignment 2 — Scene Classification

## Overview

This project compares supervised and self-supervised learning for
16-class scene classification using ResNet-18.

The best model uses SimCLR self-supervised pretraining followed by
supervised fine-tuning with 160x160 input images.

## Results

| Model | Validation Accuracy | Test Accuracy |
|---|---:|---:|
| Supervised ResNet-18 | 75.21% | 74.25% |
| SimCLR + Linear Probe | 65.21% | Not evaluated |
| SimCLR + Fine-tuning | 75.62% | Not evaluated |
| SimCLR + Regularization | 76.25% | 74.50% |
| SimCLR + MixUp | 76.04% | Not evaluated |
| SimCLR + High-Resolution Fine-tuning | 82.50% | 76.50% |

Best test accuracy: **76.50% (306/400)**

## Model

- Architecture: ResNet-18
- Pretraining: SimCLR contrastive learning
- SSL pretraining epochs: 80
- Fine-tuning epochs: 40
- Input resolution: 160x160
- Batch size: 32
- Optimizer: AdamW
- Learning rate: 0.0005
- Encoder learning-rate factor: 0.1
- Weight decay: 0.0001
- Label smoothing: 0.1
- Validation split: 20%, stratified
- Random seed: 42

No externally pretrained weights were used.

## Installation

Use a Python environment compatible with the installed PyTorch and
torchvision versions.

Install dependencies:

    python -m pip install -r requirements.txt

For GPU acceleration, install a CUDA-compatible PyTorch build.

## Dataset

Place the dataset folders in the project root:

    train/
    test/
    test2/

The training set contains 2,400 labeled images across 16 classes.
The labeled test set contains 400 images.
The unlabeled test2 set contains 400 images.

Dataset files are not included in this repository.

## Reproduce the Reported Test Accuracy

The best trained checkpoint is:

    results/finetune_v4.pt

Run:

    python ITCS_6169_Assignment2_Reed.py --stage evaluate --model finetune_v4

Expected output:

    finetune_v4 TEST accuracy: 76.50% (306/400), loss 0.7608

## Generate Unlabeled Test Predictions

Run:

    python ITCS_6169_Assignment2_Reed.py --stage predict --model finetune_v4 --batch-size 32

Predictions are written to:

    results/finetune_v4_test2_predictions.csv

The CSV contains filename, predicted_class, and confidence.

## Reproduce Final Fine-Tuning

The final fine-tuning command is:

    python ITCS_6169_Assignment2_Reed.py --stage finetune_v4 --size 160 --batch-size 32 --epochs 40 --lr 0.0005 --encoder-lr-factor 0.1 --weight-decay 0.0001 --label-smoothing 0.1

This stage requires the previously trained SimCLR encoder checkpoint,
results/ssl_encoder.pt. Run the SSL pretraining stage first when
reproducing training from scratch.

## Error Analysis

Validation error analysis can be reproduced with:

    python ITCS_6169_Assignment2_Reed.py --stage analyze --batch-size 32

The model's most difficult categories included Industrial,
OpenCountry, and LivingRoom.

## Limitations

The final model achieved higher validation accuracy than test
accuracy. Visually similar scene categories remained difficult to
distinguish.

The final high-resolution configuration changed multiple training
settings, so its improvement cannot be attributed to resolution alone.
