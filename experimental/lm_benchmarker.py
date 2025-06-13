import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd

from os import sys
# Ensure the custom optimizers directory is in the path
sys.path.append('../.')

from custom_optimizers import ElementWiseDecay  

# -- Hyperparameters --
seq_length         = 100
batch_size         = 64
embed_size         = 128
hidden_size        = 256
num_layers         = 1
learning_rate      = 1e-2
stage1_target_loss  = 1.6
stage2_target_loss  = 1.6
device             = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -- New params for partial load and reporting --
max_chars          = 2_000_000     # only read first 2M chars
report_every       = 500           # report performance every 500 batches

# -- Paths to your EuroParl text files --
path_en = "/scratch/ar2342/datasets/europarl_v10/europarl-v10.en.txt"
path_fr = "/scratch/ar2342/datasets/europarl_v10/europarl-v10.fr.txt"
assert os.path.isfile(path_en), f"Cannot find {path_en}"
assert os.path.isfile(path_fr), f"Cannot find {path_fr}"

# -- 1) Build character vocabulary over both (partial) files --
def build_vocab(paths, max_chars=None):
    chars = set()
    for p in paths:
        with open(p, 'r', encoding='utf-8') as f:
            count = 0
            for line in f:
                if max_chars is not None and count >= max_chars:
                    break
                chars.update(line)
                count += len(line)
    chars = sorted(chars)
    stoi = {ch:i for i,ch in enumerate(chars)}
    itos = {i:ch for ch,i in stoi.items()}
    return stoi, itos

stoi, itos = build_vocab([path_en, path_fr], max_chars=max_chars)
vocab_size = len(stoi)
print(f"Vocab size: {vocab_size}")

# -- 2) Dataset with partial‐load support --
class CharDataset(Dataset):
    def __init__(self, path, stoi, seq_length, max_chars=None):
        self.stoi = stoi
        self.seq_length = seq_length
        text = open(path, 'r', encoding='utf-8').read()
        if max_chars is not None:
            text = text[:max_chars]
        data = [stoi[ch] for ch in text if ch in stoi]
        self.data = torch.tensor(data, dtype=torch.long)

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_length]
        y = self.data[idx + 1 : idx + 1 + self.seq_length]
        return x, y

# build EN/FR datasets (with 10% held‐out for validation)
full_en     = CharDataset(path_en, stoi, seq_length, max_chars=max_chars)
val_size    = int(0.1 * len(full_en))
train_en, val_en = random_split(full_en, [len(full_en)-val_size, val_size])

full_fr     = CharDataset(path_fr, stoi, seq_length, max_chars=max_chars)
val_size_fr = int(0.1 * len(full_fr))
train_fr, val_fr = random_split(full_fr, [len(full_fr)-val_size_fr, val_size_fr])

loader_en_train = DataLoader(train_en, batch_size=batch_size, shuffle=True,  drop_last=True)
loader_en_val   = DataLoader(val_en,   batch_size=batch_size, shuffle=False, drop_last=True)
loader_fr_train = DataLoader(train_fr, batch_size=batch_size, shuffle=True,  drop_last=True)
loader_fr_val   = DataLoader(val_fr,   batch_size=batch_size, shuffle=False, drop_last=True)

# -- 3) Model definition --
class CharRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn   = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc    = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h=None):
        e = self.embed(x)
        out, h = self.rnn(e, h)
        logits = self.fc(out)
        return logits, h

model     = CharRNN(vocab_size, embed_size, hidden_size, num_layers).to(device)
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
optimizer = ElementWiseDecay(model.parameters(), lr=learning_rate, alpha=0)  # Using custom optimizer

# -- 4) Helpers: training with inter‐step reporting, early‐exit & evaluation --
def evaluate(loader):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_chars = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            B, L, V = logits.shape
            loss = criterion(logits.view(-1, V), y.view(-1))
            total_loss += loss.item() * B
            preds = logits.argmax(dim=2)
            total_correct += (preds == y).sum().item()
            total_chars += B * L
    return total_correct/total_chars, total_loss/len(loader.dataset)

def train_one_epoch(loader, val_loaders=None, phase="", epoch=0,
                    target_name=None, target_loss=None):
    """
    Returns: (avg_train_loss, hit_target_flag)
    - target_name: key in val_loaders whose loss we watch
    - target_loss: threshold to trigger early exit
    """
    model.train()
    total_loss = 0
    # flag to let caller know we hit the target
    hit = False

    for step, (x, y) in enumerate(loader, start=1):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)

        # --- Inter‐step reporting & early‐exit check ---
        if val_loaders and step % report_every == 0:
            # run all validations
            vlosses = {}
            stats = []
            for name, vloader in val_loaders.items():
                acc, vloss = evaluate(vloader)
                vlosses[name] = vloss
                stats.append(f"{name} val-acc: {acc:.4f}, val-loss: {vloss:.4f}")
            print(f"[{phase}][Epoch {epoch}][Step {step}] " + " | ".join(stats))
            model.train()

            # did we hit our exit condition?
            if target_name is not None \
               and target_name in vlosses \
               and vlosses[target_name] <= target_loss:
                print(f"→ Early stopping: {target_name} loss "
                      f"{vlosses[target_name]:.4f} ≤ {target_loss:.4f}")
                hit = True
                break

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, hit


# -- 5) Training loops with continuous monitoring & logging + early exit --

records = []

# Stage 1: English
epoch = 0
while True:
    epoch += 1
    train_loss, stop_now = train_one_epoch(
        loader_en_train,
        val_loaders={"EN": loader_en_val},
        phase="Stage1-EN",
        epoch=epoch,
        target_name="EN",
        target_loss=stage1_target_loss
    )
    acc_en, loss_en = evaluate(loader_en_val)
    print(f"[Stage1][Epoch {epoch}] COMPLETE → EN val-acc: {acc_en:.4f}, val-loss: {loss_en:.4f}")
    records.append({
        "phase":      "stage1",
        "epoch":      epoch,
        "train_loss": train_loss,
        "acc_en":     acc_en,
        "loss_en":    loss_en,
        "acc_fr":     None,
        "loss_fr":    None
    })
    if stop_now or loss_en <= stage1_target_loss:
        print(f"→ Finished Stage1 at epoch {epoch}")
        break

# Stage 2: French (monitor both EN & FR, but only stop on FR)
epoch2 = 0
while True:
    epoch2 += 1
    train_loss, stop_now = train_one_epoch(
        loader_fr_train,
        val_loaders={"EN": loader_en_val, "FR": loader_fr_val},
        phase="Stage2-FR",
        epoch=epoch2,
        target_name="FR",
        target_loss=stage2_target_loss
    )
    acc_en, loss_en = evaluate(loader_en_val)
    acc_fr, loss_fr = evaluate(loader_fr_val)
    print(
        f"[Stage2][Epoch {epoch2}] COMPLETE → "
        f"EN val-acc: {acc_en:.4f}, EN val-loss: {loss_en:.4f} | "
        f"FR val-acc: {acc_fr:.4f}, FR val-loss: {loss_fr:.4f}"
    )
    records.append({
        "phase":      "stage2",
        "epoch":      epoch2,
        "train_loss": train_loss,
        "acc_en":     acc_en,
        "loss_en":    loss_en,
        "acc_fr":     acc_fr,
        "loss_fr":    loss_fr
    })
    if stop_now or loss_fr <= stage2_target_loss:
        print(f"→ Finished Stage2 at epoch {epoch2}")
        break

# -- 6) Save & display metrics --
df = pd.DataFrame(records)
csv_path = "char_rnn_europarl_metrics.csv"
df.to_csv(csv_path, index=False)
print("\nAll logged metrics:")
print(df.to_string(index=False))
print(f"\nMetrics saved to {csv_path}")

