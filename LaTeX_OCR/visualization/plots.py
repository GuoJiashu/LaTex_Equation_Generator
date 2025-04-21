# visualization/plots.py
# Drawing tools (Loss curve, confusion matrix, feature map)
# Author: Jiashu Guo
# Date: 2025-04-21

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# --- Loss Curve ---

def plot_loss_curves(train_losses, val_losses, save_path="loss_curve.png"):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Saved loss curve to {save_path}")

# --- feature maps ---

def show_feature_maps(tensor, title="Feature Map", max_channels=4):
    n = min(max_channels, tensor.shape[1])
    fig, axes = plt.subplots(1, n, figsize=(3*n, 3))
    for i in range(n):
        axes[i].imshow(tensor[0, i].detach().cpu(), cmap='gray')
        axes[i].set_title(f"{title} | Channel {i}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

# --- max activation heatmap ---

def show_max_activation(feature_map, title="Max Activation"):
    max_activations = torch.max(feature_map[0], dim=0).values.detach().cpu()
    plt.figure(figsize=(6, 4))
    plt.imshow(max_activations, cmap='hot')
    plt.colorbar()
    plt.title(title)
    plt.axis('off')
    plt.show()

# --- Confusion Matrix ---

def plot_confusion_matrix(true_labels, pred_labels, top_k_tokens=30):
    """Draw the confusion matrix (display top_k high-frequency tokens)"""
    from collections import Counter

    all_tokens = Counter(true_labels + pred_labels)
    most_common = [tok for tok, _ in all_tokens.most_common(top_k_tokens)]
    most_common_set = set(most_common)

    def map_tok(tok):
        return tok if tok in most_common_set else 'other'

    mapped_true = [map_tok(tok) for tok in true_labels]
    mapped_pred = [map_tok(tok) for tok in pred_labels]

    labels_sorted = sorted(set(mapped_true + mapped_pred))
    cm = confusion_matrix(mapped_true, mapped_pred, labels=labels_sorted)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels_sorted, yticklabels=labels_sorted)
    plt.xlabel("Predicted Token")
    plt.ylabel("Ground Truth Token")
    plt.title(f"Confusion Matrix (Top {top_k_tokens} Tokens + 'other')")
    plt.tight_layout()
    plt.show()

    return cm, labels_sorted
