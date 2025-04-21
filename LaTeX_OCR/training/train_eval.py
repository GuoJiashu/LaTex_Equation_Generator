# training/train_eval.py
# Traning and Evaluation Module
# Author: Jiashu Guo
# Date: 2025-04-21

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from training.loss import create_loss_function
from training.optimizer import get_optimizer, get_scheduler
from utils_module.utils import generate_square_subsequent_mask
from torch.utils.tensorboard import SummaryWriter

def train_one_epoch(model, dataloader, optimizer, scheduler, loss_fn, device):
    model.train()
    total_loss = 0

    for images, labels, _ in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        tgt_input = labels[:, :-1]
        tgt_output = labels[:, 1:]
        tgt_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(device)

        optimizer.zero_grad()
        logits = model(images, tgt_input, tgt_mask)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

@torch.no_grad()
def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0

    for images, labels, _ in tqdm(dataloader, desc="Evaluating"):
        images, labels = images.to(device), labels.to(device)

        tgt_input = labels[:, :-1]
        tgt_output = labels[:, 1:]
        tgt_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(device)

        logits = model(images, tgt_input, tgt_mask)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))

        total_loss += loss.item()

    return total_loss / len(dataloader)

def fit(model, train_loader, val_loader, token2idx, epochs=10, lr=1e-4, device="cuda"):
    optimizer = get_optimizer(model, lr)
    loss_fn = create_loss_function(token2idx, smoothing=0.1)

    num_training_steps = epochs * len(train_loader)
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_scheduler(optimizer, num_warmup_steps, num_training_steps)

    writer = SummaryWriter(log_dir="runs/im2latex")
    best_val_loss = float('inf')

    train_losses = []
    val_losses = []

    for epoch in range(1, epochs+1):
        print(f"\nEpoch {epoch}/{epochs}")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, loss_fn, device)
        val_loss = evaluate(model, val_loader, loss_fn, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")
            print("Saved new best model!")

    writer.close()

    # Loss Curve Plotting
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_curve.png")
    plt.show()
