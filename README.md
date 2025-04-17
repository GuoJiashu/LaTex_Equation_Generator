# LaTex_Equations_Generator

# 🖋️ Handwritten LaTeX Recognition (Im2LaTeX) - Full Project

基于 PyTorch 实现的完整手写公式识别系统，支持数据预处理、模型训练、推理评估、错误分析、混淆矩阵可视化等一系列功能。

---

## 📦 项目结构

- `dataset_preprocessing.py` ：图像与Latex公式的处理、编码
- `model.py` ：Encoder-Decoder 结构，带 Positional Encoding 和 Transformer Decoder
- `train.py` ：训练、验证、测试流程封装
- `evaluate.py` ：支持 Token-level、Sentence-level、Edit Distance、多种评估指标
- `beam_search_decode.py` ：Beam Search 解码器推理
- `analysis_tools.py` ：生成混淆矩阵、错误分析、Top混淆对查找
- `export_results.py` ：推理结果导出CSV、绘图保存

（实际你是放在一个大Notebook或者.py里，但可以这么模块化理解✨）

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install torch torchvision pandas scikit-learn matplotlib seaborn tqdm pillow python-Levenshtein

