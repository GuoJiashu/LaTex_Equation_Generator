# inference/beam_search.py
# Beam Search decoder
# Author: Jiashu Guo
# Date: 2025-04-21

import torch
from utils_module.utils import generate_square_subsequent_mask

@torch.no_grad()
def beam_search_decode(model, image, token2idx, idx2token, beam_width=3, max_len=256, device="cuda", length_penalty_alpha=0.6):
    model.eval()

    if not image.is_cuda:
        image = image.to(device)

    memory = model.encoder(image.unsqueeze(0))  # One image at a time
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
