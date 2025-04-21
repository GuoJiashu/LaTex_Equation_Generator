# data/preprocess.py
# Latex OCR Data Preprocessing Script
# Author: Jiashu Guo/ Junhao Fu
# Date: 2025-04-21

import re
import numpy as np
import torch
from PIL import Image, ImageOps

def smart_clean_latex(code: str) -> str:
    """Clean Latex code by removing unnecessary spaces and preserving commands."""
    preserved = re.findall(r'(\\[a-zA-Z]+)\s+', code)
    for cmd in preserved:
        code = code.replace(cmd + ' ', f'{cmd}<<<SPACE>>>')
    code = code.replace(' ', '')
    code = code.replace('<<<SPACE>>>', ' ')
    return code

def mixed_tokenize_latex(code: str, command_set: set) -> list:
    """Tokenize LaTeX code into a list of tokens."""
    tokens = []
    i = 0
    while i < len(code):
        if code[i] == '\\':
            j = i + 1
            while j < len(code) and code[j].isalpha():
                j += 1
            cmd = code[i:j]
            if cmd in command_set:
                tokens.append(cmd)
                i = j
                continue
        tokens.append(code[i])
        i += 1
    return tokens

def encode_label_mixed(code: str, token2idx: dict, command_set: set, max_len: int) -> np.ndarray:
    """Convert LaTeX code to a sequence of indices."""
    tokens = ['<s>'] + mixed_tokenize_latex(code, command_set)[:max_len-2] + ['</s>']
    label = np.zeros(max_len, dtype=np.int32)
    for t, token in enumerate(tokens):
        label[t] = token2idx.get(token, 0)
    return label

def preprocess_image(image_path: str, target_size=(512, 128)) -> torch.Tensor:
    """scaling + filling + normalization + expansion 3 channels"""
    img = Image.open(image_path).convert('L')
    img.thumbnail(target_size, Image.Resampling.LANCZOS)
    delta_w = target_size[0] - img.size[0]
    delta_h = target_size[1] - img.size[1]
    padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)
    img = ImageOps.expand(img, padding, fill=255)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.repeat(img_array, 3, axis=0)
    return torch.from_numpy(img_array).float()
