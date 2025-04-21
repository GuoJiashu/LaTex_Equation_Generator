# train.py
# 训练主程序入口
# Author: YourName
# Date: 2025-04-21

import torch
from config import config
from data_process.dataset import get_data_loaders
from model_build.model import Im2LatexModel
from training.train_eval import fit
import pickle
import os

def main():
    print("Preparing data loaders...")
    train_loader, val_loader, _ = get_data_loaders(config)

    # 加载token2idx
    with open(os.path.join(config.token_dict_dir, "token2idx.pkl"), "rb") as f:
        token2idx = pickle.load(f)
    config.token2idx = token2idx
    config.vocab_size = len(token2idx)

    print("Building model...")
    model = Im2LatexModel(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        max_seq_len=config.max_seq_len
    ).to(config.device)

    print(f"Starting training for {config.epochs} epochs...")
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        token2idx=config.token2idx,
        epochs=config.epochs,
        lr=config.lr,
        device=config.device
    )

if __name__ == "__main__":
    main()
