# data/dataset.py
# Encapsulation of Dataset and DataLoader
# Author: Jiashu Guo/ Junhao Fu
# Date: 2025-04-21

import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import pandas as pd
from data_process.preprocess import preprocess_image, encode_label_mixed, smart_clean_latex, mixed_tokenize_latex
from sklearn.model_selection import train_test_split
import pickle

# --- Dataset Class ---

class LatexDataset(Dataset):
    def __init__(self, image_dir, image_names, latex_codes, max_len, token2idx, command_set, augment=False):
        self.image_dir = image_dir
        self.image_names = image_names
        self.latex_codes = latex_codes
        self.max_len = max_len
        self.token2idx = token2idx
        self.command_set = command_set
        self.augment = augment

        self.transform = T.Compose([
            T.ToPILImage(),
            T.RandomRotation(degrees=2),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, f"{self.image_names[idx]}.bmp")
        img = preprocess_image(img_path)

        if self.augment:
            img = self.transform(img)

        label = encode_label_mixed(self.latex_codes[idx], self.token2idx, self.command_set, self.max_len)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return img, label_tensor

# --- collate function ---

def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)
    return images, labels, [len(l) for l in labels]

# --- Load All DataLoader ---

def get_data_loaders(config):
    # Load csv
    csv_data = pd.read_csv(config.csv_file)
    latex_codes_raw = csv_data['Column2'].values.tolist()
    image_names = csv_data['Column1'].values.tolist()

    # Cleaning
    latex_codes = [smart_clean_latex(str(code)) for code in latex_codes_raw]

    # Extract command tokens
    command_tokens = set()
    for code in latex_codes:
        command_tokens.update([tok for tok in mixed_tokenize_latex(code, command_tokens) if tok.startswith('\\')])

    # Load token2idx
    with open(os.path.join(config.data_dir, "token_dicts/token2idx.pkl"), "rb") as f:
        token2idx = pickle.load(f)
    config.token2idx = token2idx
    config.vocab_size = len(token2idx)

    # Split data
    train_imgs, tep_imgs, train_codes, tep_codes = train_test_split(image_names, latex_codes, test_size=0.3, random_state=42)
    val_imgs, test_imgs, val_codes, test_codes = train_test_split(tep_imgs, tep_codes, test_size=0.5, random_state=41)

    # Dataset
    train_dataset = LatexDataset(config.handwritten_equations, train_imgs, train_codes, config.max_seq_len, token2idx, command_tokens, augment=True)
    val_dataset = LatexDataset(config.handwritten_equations, val_imgs, val_codes, config.max_seq_len, token2idx, command_tokens, augment=False)
    test_dataset = LatexDataset(config.handwritten_equations, test_imgs, test_codes, config.max_seq_len, token2idx, command_tokens, augment=False)

    # Dataloader
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader
