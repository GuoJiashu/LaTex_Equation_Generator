# models/decoder.py
# Transformer Decoder
# Author: Jiashu Guo
# Date: 2025-04-21

import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model=384, nhead=8, num_layers=8, max_len=256, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len*2, d_model))  # Learnable
        self.output_layer = nn.Linear(d_model, vocab_size)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        nn.init.normal_(self.pos_embedding, std=0.02)

    def forward(self, tgt_input, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None):
        tgt_emb = self.token_embedding(tgt_input) + self.pos_embedding[:, :tgt_input.size(1), :]
        memory = memory.reshape(memory.size(0), -1, memory.size(-1))  # flatten spatial
        output = self.transformer_decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        return self.output_layer(output)
