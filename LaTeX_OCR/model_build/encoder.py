# models/encoder.py
# ResNet34 Backbone + LSTM
# Author: Jiashu Guo
# Date: 2025-04-21

import torch
import torch.nn as nn
from torchvision.models import resnet34
from model_build.positional_encoding import PositionalEncoding2D

class Encoder(nn.Module):
    def __init__(self, d_model=384):
        super().__init__()
        base_cnn = resnet34(pretrained=True)
        self.backbone = nn.Sequential(*list(base_cnn.children())[:6])
        self.project = nn.Conv2d(128, d_model, kernel_size=1)
        self.row_encoder = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model//2,
            batch_first=True,
            bidirectional=True
        )
        self.pe = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        feat = self.project(feat)
        B, C, H, W = feat.shape

        if self.pe is None:
            self.pe = PositionalEncoding2D(C, H, W).to(x.device)
        feat = self.pe(feat)

        feat = feat.permute(0, 2, 3, 1).contiguous().view(B * H, W, C)
        feat, _ = self.row_encoder(feat)
        feat = feat.view(B, H, W, C)
        return feat
