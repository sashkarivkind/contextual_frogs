from os import sys
# Ensure the custom optimizers directory is in the path
sys.path.append('../.')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import pandas as pd
from custom_optimizers import ElementWiseDecay  
# -- Hyperparameters --
batch_size = 64
learning_rate = 1e-3

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -- Data preparation --
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_full = datasets.MNIST('.', train=True, download=True, transform=transform)
test_full  = datasets.MNIST('.', train=False, download=True, transform=transform)

# Filter indices
idx_train_0_4 = [i for i, t in enumerate(train_full.targets) if t < 5]
idx_train_5_9 = [i for i, t in enumerate(train_full.targets) if t >= 5]
idx_test_0_4  = [i for i, t in enumerate(test_full.targets)  if t < 5]
idx_test_5_9  = [i for i, t in enumerate(test_full.targets)  if t >= 5]

train_0_4 = DataLoader(Subset(train_full, idx_train_0_4), batch_size=batch_size, shuffle=True)
train_5_9 = DataLoader(Subset(train_full, idx_train_5_9), batch_size=batch_size, shuffle=True)
test_0_4  = DataLoader(Subset(test_full,  idx_test_0_4),  batch_size=batch_size, shuffle=False)
test_5_9  = DataLoader(Subset(test_full,  idx_test_5_9),  batch_size=batch_size, shuffle=False)

# -- Model definition --
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        return self.fc2(x)

class SimpleMLP(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input
        x = F.relu(self.fc1(x))
        return self.fc2(x)

model = SimpleCNN().to(device)
# model = SimpleMLP().to(device)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = optim.SGD(model.parameters(), lr=learning_rate)
optimizer = ElementWiseDecay(model.parameters(), lr=learning_rate, alpha=100)  # Using custom optimizer
criterion = nn.CrossEntropyLoss()

# -- Helper functions --
def train_one_epoch(loader):
    model.train()
    total_loss = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.size(0)
    return total_loss / len(loader.dataset)

def evaluate(loader):
    model.eval()
    correct = 0
    total_loss = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    acc = correct / len(loader.dataset)
    avg_loss = total_loss / len(loader.dataset)
    return acc, avg_loss

# -- Training & logging --
records = []

# Stage 1: train on digits 0-4 until ≥95% accuracy
epoch = 0
while True:
    epoch += 1
    train_loss = train_one_epoch(train_0_4)
    val_acc, val_loss = evaluate(test_0_4)
    print(f"[Stage 1][Epoch {epoch}] Train loss: {train_loss:.4f} | Val0-4 acc: {val_acc:.4f}, loss: {val_loss:.4f}")
    records.append({
        'phase': 'stage1',
        'epoch': epoch,
        'acc_0_4': val_acc,
        'loss_0_4': val_loss,
        'acc_5_9': None,
        'loss_5_9': None
    })
    if val_acc >= 0.95:
        print(f"Reached 95% validation accuracy on 0-4 at epoch {epoch}.")
        break

# Stage 2: train on digits 5-9 until ≥95% accuracy on 5-9
stage2_epoch = 0
while True:
    stage2_epoch += 1
    train_loss_5_9 = train_one_epoch(train_5_9)
    acc0, loss0 = evaluate(test_0_4)
    acc5, loss5 = evaluate(test_5_9)
    print(f"[Stage 2][Epoch {stage2_epoch}] Train5-9 loss: {train_loss_5_9:.4f} | Test0-4 acc: {acc0:.4f}, loss: {loss0:.4f} | Test5-9 acc: {acc5:.4f}, loss: {loss5:.4f}")
    records.append({
        'phase': 'stage2',
        'epoch': stage2_epoch,
        'acc_0_4': acc0,
        'loss_0_4': loss0,
        'acc_5_9': acc5,
        'loss_5_9': loss5
    })
    if acc5 >= 0.95:
        print(f"Reached 95% validation accuracy on 5-9 at Stage 2 epoch {stage2_epoch}.")
        break

# -- Save & display results --
df = pd.DataFrame(records)
csv_path = 'metrics.csv'
df.to_csv(csv_path, index=False)

print("\nAll results:")
print(df.to_string(index=False))
print(f"\nMetrics saved to {csv_path}")
