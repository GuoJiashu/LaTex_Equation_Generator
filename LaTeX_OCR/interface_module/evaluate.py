# inference/evaluate.py
# Various inference metrics (accuracy rate, edit distance, BLEU)
# Author: Jiashu Guo
# Date: 2025-04-21

import torch
import numpy as np
import Levenshtein
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from interface_module.beam_search import beam_search_decode

@torch.no_grad()
def evaluate_model_with_beam(model, dataset, token2idx, idx2token, num_samples, beam_width=3, max_len=256, visualize=False):
    model.eval()
    correct = 0
    total = 0

    device = next(model.parameters()).device

    def tokens_to_str(tokens):
        return ''.join(t for t in tokens if t not in ['<s>', '</s>', '<pad>'])

    for i in range(num_samples):
        image, label = dataset[i]
        image = image.to(device)

        pred_tokens = beam_search_decode(model, image, token2idx, idx2token, beam_width, max_len, device=device)
        true_tokens = [idx2token[idx.item()] for idx in label if idx.item() != token2idx['<pad>']]

        pred_str = tokens_to_str(pred_tokens)
        true_str = tokens_to_str(true_tokens)

        if pred_str == true_str:
            correct += 1
        total += 1

    acc = correct / total
    print(f"Beam Search Exact Match Accuracy: {acc*100:.2f}%")
    return acc

@torch.no_grad()
def evaluate_token_level_accuracy(model, dataset, token2idx, idx2token, num_samples=50, beam_width=3, max_len=256):
    model.eval()
    correct_tokens = 0
    total_tokens = 0

    device = next(model.parameters()).device

    def strip(tokens):
        return [t for t in tokens if t not in ['<pad>', '<s>', '</s>']]

    for i in range(num_samples):
        image, label = dataset[i]
        image = image.to(device)

        pred_tokens = beam_search_decode(model, image, token2idx, idx2token, beam_width, max_len, device=device)
        true_tokens = [idx2token[idx.item()] for idx in label]

        pred = strip(pred_tokens)
        true = strip(true_tokens)

        for p, t in zip(pred, true):
            total_tokens += 1
            if p == t:
                correct_tokens += 1

        total_tokens += abs(len(pred) - len(true))

    accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
    print(f"Token-level Accuracy: {accuracy*100:.2f}%")
    return accuracy

@torch.no_grad()
def evaluate_edit_distance(model, dataset, token2idx, idx2token, num_samples=50, beam_width=3, max_len=256):
    model.eval()
    total_ned = 0.0
    count = 0

    device = next(model.parameters()).device

    def strip(tokens):
        return [t for t in tokens if t not in ['<pad>', '<s>', '</s>']]

    for i in range(num_samples):
        image, label = dataset[i]
        image = image.to(device)

        pred_tokens = beam_search_decode(model, image, token2idx, idx2token, beam_width, max_len, device=device)
        true_tokens = [idx2token[idx.item()] for idx in label]

        pred = strip(pred_tokens)
        true = strip(true_tokens)

        pred_str = ' '.join(pred)
        true_str = ' '.join(true)

        if len(true_str.strip()) == 0:
            continue

        lev_dist = Levenshtein.distance(pred_str, true_str)
        max_len_str = max(len(pred_str), len(true_str))
        ned = 1 - lev_dist / max_len_str
        total_ned += ned
        count += 1

    avg_ned = total_ned / count if count > 0 else 0.0
    print(f"Normalized Edit Distance (NED): {avg_ned*100:.2f}%")
    return avg_ned

@torch.no_grad()
def evaluate_bleu_score(model, dataset, token2idx, idx2token, num_samples=50, beam_width=3, max_len=256):
    model.eval()
    smoothing_fn = SmoothingFunction().method4
    bleu_scores = []

    device = next(model.parameters()).device

    def strip(tokens):
        return [t for t in tokens if t not in ['<pad>', '<s>', '</s>']]

    for i in range(num_samples):
        image, label = dataset[i]
        image = image.to(device)

        pred_tokens = beam_search_decode(model, image, token2idx, idx2token, beam_width, max_len, device=device)
        true_tokens = [idx2token[idx.item()] for idx in label]

        pred = strip(pred_tokens)
        true = strip(true_tokens)

        if len(true) == 0:
            continue

        bleu = sentence_bleu(
            [true], pred,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoothing_fn
        )
        bleu_scores.append(bleu)

    avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
    print(f"BLEU Score: {avg_bleu*100:.2f}%")
    return avg_bleu
