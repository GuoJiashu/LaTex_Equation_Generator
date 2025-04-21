# config.py
# Global configuration parameters
# Author: YourName
# Date: 2025-04-21

import os
import torch

class Config:
    """Configuration management class"""
    data_dir = r"C:\Users\13658\Desktop\LaTex_Code_Generator"   # 你的数据路径
    handwritten_equations = os.path.join(data_dir, "Handwritten_equations")
    csv_file = os.path.join(data_dir, "caption_data.csv")
    token_dict_dir = os.path.join(data_dir, "token_dicts")

    batch_size = 32
    lr = 1e-4
    epochs = 50
    d_model = 384
    max_seq_len = 256

    vocab_size = None
    token2idx = None
    idx2token = None

    device = "cuda" if torch.cuda.is_available() else "cpu"

config = Config()
