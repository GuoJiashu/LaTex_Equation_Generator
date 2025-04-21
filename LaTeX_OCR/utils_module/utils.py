# utils/utils.py
# General widget functions
# Author: Jiashu Guo
# Date: 2025-04-21

import torch

def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """Generate the causal mask of the Transformer (mask the future position)"""
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def strip_special_tokens(tokens: list, special_tokens=None) -> list:
    """Remove these special marks such as <pad>, <s>, and </s>"""
    if special_tokens is None:
        special_tokens = ['<pad>', '<s>', '</s>']
    return [t for t in tokens if t not in special_tokens]

def tokens_to_string(tokens: list) -> str:
    """Convert the token list to a string"""
    return ''.join(tokens)
