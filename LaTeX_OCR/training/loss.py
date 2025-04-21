# training/loss.py
# Loss Function (label smoothing CrossEntropy)
# Author: Jiashu Guo
# Date: 2025-04-21

import torch
import torch.nn as nn

def create_loss_function(token2idx: dict, smoothing=0.1):
    """Create a labeled smooth cross-entropy loss"""
    ignore_index = token2idx['<pad>']
    return nn.CrossEntropyLoss(ignore_index=ignore_index, label_smoothing=smoothing)
