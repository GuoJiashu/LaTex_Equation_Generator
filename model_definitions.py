import math
import numpy as np
from PIL import Image, ImageOps
import pickle

import torch
import torch.nn as nn
from torchvision.models import resnet34

# === 加载词表 ===
with open("token_dicts/token2idx.pkl", "rb") as f:
    token2idx = pickle.load(f)
with open("token_dicts/idx2token.pkl", "rb") as f:
    idx2token = pickle.load(f)

vocab_size = len(token2idx)
max_seq_length = 256  # 固定长度，训练时也使用了这个长度

# === 模型结构定义 ===
class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, height, width):
        super().__init__()
        if d_model % 4 != 0:
            raise ValueError("d_model must be divisible by 4")

        self.d_model = d_model
        self.height = height
        self.width = width

        pe = torch.zeros(d_model, height, width)

        d_model_half = d_model // 2
        div_term = torch.exp(torch.arange(0., d_model_half, 2) * -(math.log(10000.0) / d_model_half))

        pos_w = torch.arange(0., width).unsqueeze(1)  # [W, 1]
        pos_h = torch.arange(0., height).unsqueeze(1)  # [H, 1]

        # 横向编码（宽度方向）
        pe[0:d_model_half:2, :, :] = torch.sin(pos_w * div_term).T.unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model_half:2, :, :] = torch.cos(pos_w * div_term).T.unsqueeze(1).repeat(1, height, 1)

        # 纵向编码（高度方向）
        pe[d_model_half::2, :, :] = torch.sin(pos_h * div_term).T.unsqueeze(2).repeat(1, 1, width)
        pe[d_model_half+1::2, :, :] = torch.cos(pos_h * div_term).T.unsqueeze(2).repeat(1, 1, width)

        self.register_buffer('pe', pe.unsqueeze(0))  # shape: [1, d_model, H, W]

    def forward(self, x):
        return x + self.pe[:, :, :x.size(2), :x.size(3)]



class Encoder(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        base_cnn = resnet34(pretrained=True)
        self.backbone = nn.Sequential(*list(base_cnn.children())[:6])  # conv1 + bn1 + relu + maxpool + layer1, layer2

        self.project = nn.Conv2d(128, d_model, kernel_size=1)  # map channels to d_model
        self.row_encoder = nn.LSTM(input_size=d_model,
                                   hidden_size=d_model // 2,
                                   batch_first=True,
                                   bidirectional=True)

        self.pe = None  # lazy initialization for Positional Encoding

    def forward(self, x):
        feat = self.backbone(x)           # [B, 128, H, W]
        feat = self.project(feat)         # [B, d_model, H, W]
        B, C, H, W = feat.shape

        if self.pe is None:
            self.pe = PositionalEncoding2D(C, H, W).to(x.device)
        feat = self.pe(feat)

        # Run Row-wise LSTM
        feat = feat.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        feat = feat.view(B * H, W, C)                 # [B*H, W, C]
        feat, _ = self.row_encoder(feat)
        feat = feat.view(B, H, W, C)

        return feat  # [B, H, W, D]

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=8, max_len=256, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len * 2, d_model))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, vocab_size)

        nn.init.normal_(self.pos_embedding, std=0.02)

    def forward(self, tgt_input, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None):
        # tgt_input: [B, T]
        tgt_emb = self.token_embedding(tgt_input) + self.pos_embedding[:, :tgt_input.size(1), :]  # [B, T, D]

        # memory: [B, H, W, D] -> [B, HW, D]
        memory = memory.reshape(memory.size(0), -1, memory.size(-1))


        # Decoder output
        output = self.transformer_decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        return self.output_layer(output)  # [B, T, vocab_size]

class Im2LatexModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, max_seq_len=256):
        super().__init__()
        self.encoder = Encoder(d_model=d_model)
        self.decoder = Decoder(vocab_size=vocab_size, d_model=d_model, max_len=max_seq_len)
        self.max_seq_len = max_seq_len

    def forward(self, images, tgt_input, tgt_mask=None):
        enc_output = self.encoder(images)  # [B, H, W, D]
        logits = self.decoder(tgt_input, enc_output, tgt_mask=tgt_mask)
        return logits  # [B, T, vocab_size]

# === 生成 decoder mask ===
def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# === Beam Search 解码函数 ===
def beam_search_decode(model, image, token2idx, idx2token, beam_width=3, max_len=256, device="cuda", length_penalty_alpha=0.6):
    model.eval()

    if not image.is_cuda:
        image = image.to(device)

    with torch.no_grad():
        memory = model.encoder(image.unsqueeze(0))
        memory = memory.expand(beam_width, *memory.shape[1:])

        sequences = [[token2idx['<s>']] for _ in range(beam_width)]
        scores = torch.zeros(beam_width, device=device)
        completed = []

        for step in range(max_len):
            all_candidates = []
            for i, seq in enumerate(sequences):
                if len(seq) > 1 and seq[-1] == token2idx['</s>']:
                    # ➡️ 这里加 Length Penalty 归一化分数
                    length_penalty = (len(seq)) ** length_penalty_alpha
                    completed.append((seq, scores[i].item() / length_penalty))
                    continue

                tgt_input = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
                tgt_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(device)
                output = model.decoder(tgt_input, memory[i:i+1], tgt_mask)
                probs = torch.log_softmax(output[0, -1], dim=-1)

                topk_probs, topk_idx = torch.topk(probs, beam_width)
                for j in range(beam_width):
                    candidate = seq + [topk_idx[j].item()]
                    score = scores[i] + topk_probs[j]
                    all_candidates.append((candidate, score))

            # 选最好的 beam_width 个序列
            all_candidates = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
            sequences = [cand[0] for cand in all_candidates]
            scores = torch.tensor([cand[1] for cand in all_candidates], device=device)

        # 最后把未完成的序列也加进去
        for i, seq in enumerate(sequences):
            length_penalty = (len(seq)) ** length_penalty_alpha
            completed.append((seq, scores[i].item() / length_penalty))

        completed = sorted(completed, key=lambda x: x[1], reverse=True)

        best_seq = completed[0][0]
        return [idx2token[idx] for idx in best_seq[1:-1] if idx in idx2token]



# === 图像预处理函数 ===
def preprocess_image(image_path, target_size=(512, 128), auto_invert=True):
    img = Image.open(image_path).convert('L')

    if auto_invert:
        mean_pixel = np.array(img).mean()
        if mean_pixel > 127:  # 平均像素亮度，大概中间值
            img = ImageOps.invert(img)  # 🔥 只对亮图做反转

    img.thumbnail(target_size, Image.Resampling.LANCZOS)

    delta_w = target_size[0] - img.size[0]
    delta_h = target_size[1] - img.size[1]
    padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)
    img = ImageOps.expand(img, padding, fill=255)

    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.repeat(img_array, 3, axis=0)
    return torch.from_numpy(img_array).float()
