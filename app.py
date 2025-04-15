import streamlit as st
from PIL import Image
import torch

# ==== 导入你封装的模块 ====
from model_definitions import (
    Im2LatexModel,
    preprocess_image,
    beam_search_decode,
    token2idx,
    idx2token,
    max_seq_length
)

# ==== 模型加载 ====
@st.cache_resource
def load_model():
    model = Im2LatexModel(
        vocab_size=len(token2idx),
        d_model=256,
        max_seq_len=max_seq_length
    ).to("cuda")

    # 加载模型参数
    state_dict = torch.load("best_model.pt", map_location="cuda")

    # 删除 shape 不匹配的 pos_embedding
    if "decoder.pos_embedding" in state_dict:
        del state_dict["decoder.pos_embedding"]

    # 加载参数（允许部分不匹配）
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


# ==== 页面 UI ====
st.set_page_config(page_title="🧠 LaTeX Generator", layout="wide")
st.title("🖋️ 手写公式转 LaTeX Demo")
st.write("上传一张手写公式图像, AI 将自动生成对应的 LaTeX 表达式。")

uploaded_file = st.file_uploader("上传图像(支持 .bmp / .png / .jpg)", type=["bmp", "png", "jpg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="上传图像", use_container_width=True)

    with st.spinner("🧠 模型正在识别中，请稍候..."):
        model = load_model()
        input_tensor = preprocess_image(uploaded_file).to("cuda")

        tokens = beam_search_decode(
            model=model,
            image=input_tensor,
            token2idx=token2idx,
            idx2token=idx2token,
            beam_width=3
        )

        # 清洗特殊 token
        latex_code = ''.join(t for t in tokens if t not in ['<s>', '</s>', '<pad>'])

    st.success("✅ 识别完成！")
    st.code(latex_code, language="latex")
    st.latex(latex_code)
