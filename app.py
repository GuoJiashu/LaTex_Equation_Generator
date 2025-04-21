# 最最最最上面，一定是 Streamlit的第一句
import streamlit as st
st.set_page_config(page_title="🧠 Handwritten Formula to LaTeX", layout="wide")

# 再正常 import
import os
import math
import pickle
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision.models import resnet34

# === 先封装 加载词表 ===
@st.cache_resource
def load_token_dicts():
    with open("token_dicts/token2idx.pkl", "rb") as f:
        token2idx = pickle.load(f)
    with open("token_dicts/idx2token.pkl", "rb") as f:
        idx2token = pickle.load(f)
    return token2idx, idx2token

token2idx, idx2token = load_token_dicts()
vocab_size = len(token2idx)
max_seq_length = 256  # 跟训练时保持一致

# === 封装模型结构 ===
class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, height, width):
        super().__init__()
        pe = torch.zeros(d_model, height, width)
        d_model_half = d_model // 2
        div_term = torch.exp(torch.arange(0., d_model_half, 2) * -(math.log(10000.0) / d_model_half))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model_half:2, :, :] = torch.sin(pos_w * div_term).T.unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model_half:2, :, :] = torch.cos(pos_w * div_term).T.unsqueeze(1).repeat(1, height, 1)
        pe[d_model_half::2, :, :] = torch.sin(pos_h * div_term).T.unsqueeze(2).repeat(1, 1, width)
        pe[d_model_half+1::2, :, :] = torch.cos(pos_h * div_term).T.unsqueeze(2).repeat(1, 1, width)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :, :x.size(2), :x.size(3)]

class Encoder(nn.Module):
    def __init__(self, d_model=384):
        super().__init__()
        base_cnn = resnet34(pretrained=True)
        self.backbone = nn.Sequential(*list(base_cnn.children())[:6])
        self.project = nn.Conv2d(128, d_model, kernel_size=1)
        self.row_encoder = nn.LSTM(d_model, d_model//2, batch_first=True, bidirectional=True)
        self.pe = None

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.project(feat)
        B, C, H, W = feat.shape
        if self.pe is None:
            self.pe = PositionalEncoding2D(C, H, W).to(x.device)
        feat = self.pe(feat)
        feat = feat.permute(0,2,3,1).contiguous().view(B*H, W, C)
        feat, _ = self.row_encoder(feat)
        feat = feat.view(B, H, W, C)
        return feat

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model=384, nhead=8, num_layers=8, max_len=256, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len*2, d_model))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, vocab_size)
        nn.init.normal_(self.pos_embedding, std=0.02)

    def forward(self, tgt_input, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None):
        tgt_emb = self.token_embedding(tgt_input) + self.pos_embedding[:, :tgt_input.size(1), :]
        memory = memory.reshape(memory.size(0), -1, memory.size(-1))
        output = self.transformer_decoder(
            tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        return self.output_layer(output)

class Im2LatexModel(nn.Module):
    def __init__(self, vocab_size, d_model=384, max_seq_len=256):
        super().__init__()
        self.encoder = Encoder(d_model=d_model)
        self.decoder = Decoder(vocab_size, d_model, max_len=max_seq_len)

    def forward(self, images, tgt_input, tgt_mask=None):
        enc_output = self.encoder(images)
        logits = self.decoder(tgt_input, enc_output, tgt_mask)
        return logits

# 加载模型
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Im2LatexModel(vocab_size=vocab_size, d_model=384, max_seq_len=max_seq_length).to(device)
    state_dict = torch.load("best_model.pt", map_location=device)
    if "decoder.pos_embedding" in state_dict:
        del state_dict["decoder.pos_embedding"]
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

# 预处理图片
def preprocess_image(uploaded_file, target_size=(512,128), auto_invert=True):
    img = Image.open(uploaded_file).convert('L')
    if auto_invert:
        mean_pixel = np.array(img).mean()
        if mean_pixel > 127:
            img = ImageOps.invert(img)
    img.thumbnail(target_size, Image.Resampling.LANCZOS)
    delta_w = target_size[0] - img.size[0]
    delta_h = target_size[1] - img.size[1]
    padding = (delta_w//2, delta_h//2, delta_w - delta_w//2, delta_h - delta_h//2)
    img = ImageOps.expand(img, padding, fill=255)
    img_array = np.array(img, dtype=np.float32)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.repeat(img_array, 3, axis=0)
    return torch.from_numpy(img_array).float()

# Beam Search解码
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0,1)
    mask = mask.float().masked_fill(mask==0, float('-inf')).masked_fill(mask==1, float(0.0))
    return mask

@torch.no_grad()
def beam_search_decode(model, image, token2idx, idx2token, beam_width=3, max_len=256, device="cuda", length_penalty_alpha=0.6):
    model.eval()
    if not image.is_cuda:
        image = image.to(device)
    memory = model.encoder(image.unsqueeze(0))
    memory = memory.expand(beam_width, *memory.shape[1:])
    sequences = [[token2idx['<s>']] for _ in range(beam_width)]
    scores = torch.zeros(beam_width, device=device)
    completed = []
    for step in range(max_len):
        all_candidates = []
        for i, seq in enumerate(sequences):
            if len(seq) > 1 and seq[-1] == token2idx['</s>']:
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
        all_candidates = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
        sequences = [cand[0] for cand in all_candidates]
        scores = torch.tensor([cand[1] for cand in all_candidates], device=device)
    for i, seq in enumerate(sequences):
        length_penalty = (len(seq)) ** length_penalty_alpha
        completed.append((seq, scores[i].item() / length_penalty))
    completed = sorted(completed, key=lambda x: x[1], reverse=True)
    best_seq = completed[0][0]
    return [idx2token[idx] for idx in best_seq[1:-1] if idx in idx2token]

# === 页面布局 ===
st.title("🖋️ 手写公式转 LaTeX Demo")
st.write("上传一张手写公式图像，AI 将自动生成对应的 LaTeX 表达式。")

uploaded_file = st.file_uploader("✨ 上传图像 (.bmp / .png / .jpg)", type=["bmp", "png", "jpg"])

if uploaded_file:
    col1, col2 = st.columns([1,2])

    with col1:
        st.image(uploaded_file, caption="上传图像", use_container_width=True)

    with col2:
        with st.spinner('识别中...'):
            model = load_model()
            input_tensor = preprocess_image(uploaded_file).to("cuda")
            tokens = beam_search_decode(model, input_tensor, token2idx, idx2token, beam_width=3)
            latex_code = ''.join(t for t in tokens if t not in ['<s>', '</s>', '<pad>'])
        
        st.success("✅ 识别完成！")
        st.code(latex_code, language="latex")
        st.latex(latex_code)
