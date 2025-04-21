# training/optimizer.py
# Optimizer and learning rate scheduler
# Author: Jiashu Guo
# Date: 2025-04-21

import math
import torch

def get_optimizer(model, lr):
    """Create an Adam optimizer"""
    return torch.optim.Adam(model.parameters(), lr=lr)

def get_scheduler(optimizer, num_warmup_steps, num_training_steps):
    """Create a cosine learning rate scheduler with warmup"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            0.5 * (1.0 + math.cos(math.pi * (current_step - num_warmup_steps) / (num_training_steps - num_warmup_steps)))
        )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
