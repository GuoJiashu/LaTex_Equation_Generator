# LaTeX-OCR

Handwritten Mathematical Formula Recognition using CNN + Transformer Architecture.  
This project converts images of handwritten LaTeX equations into machine-readable LaTeX code.

---

## 📚 Project Overview

LaTeX-OCR is a deep learning pipeline for recognizing and converting images of handwritten mathematical expressions into LaTeX code. It uses:

- **Encoder**: ResNet34 backbone with 2D positional encoding and row-wise LSTM.
- **Decoder**: Transformer Decoder architecture.
- **Training**: Label smoothing loss, warm-up and cosine annealing scheduler.
- **Inference**: Beam Search decoding for sequence generation.

---

## 🚀 Features

- Supports handwritten mathematical formulas.
- Augmented training (random rotation, color jitter).
- Beam Search decoding for higher prediction accuracy.
- Token-level accuracy, Normalized Edit Distance (NED), and BLEU score evaluation.
- Visualization tools (loss curves, feature maps, confusion matrix).

---

## 🛠 Installation

1. Clone the repository:

```bash
git clone https://github.com/GuoJiashu/LaTex_Equation_Generator
cd latex-ocr
