# models/model.py
# Encapsulation of the overall Im2Latex model
# Author: YourName
# Date: 2025-04-21

import torch
import torch.nn as nn
from model_build.encoder import Encoder
from model_build.decoder import Decoder

class Im2LatexModel(nn.Module):
    def __init__(self, vocab_size, d_model=384, max_seq_len=256):
        super().__init__()
        self.encoder = Encoder(d_model=d_model)
        self.decoder = Decoder(
            vocab_size=vocab_size,
            d_model=d_model,
            max_len=max_seq_len
        )
        self.max_seq_len = max_seq_len

    def forward(self, images, tgt_input, tgt_mask=None):
        enc_output = self.encoder(images)  # [B, H, W, D]
        logits = self.decoder(tgt_input, enc_output, tgt_mask=tgt_mask)
        return logits
