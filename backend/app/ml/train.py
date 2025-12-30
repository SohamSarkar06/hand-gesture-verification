import os
import json
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

from model import GestureNet

# ================= PATHS =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.abspath(
    os.path.join(BASE_DIR, "../../../dataset/leapGestRecog")
)

MODEL_PATH = os.path.join(BASE_DIR, "gesture_model.pth")
LABELS_PATH = os.path.join(BASE_DIR, "labels.json")

# ================= DATASET =================
class LeapGestureDataset(Dataset):
    def __init__(self, root, transform=None):
        self.samples = []
        self.transform = transform

        for subject in os.listdir(root):
            subject_path = os.path.join(root, subject)
            if not os.path.isdir(subject_path):
                continue

            for gesture in os.listdir(subject_path):
                gesture_path = os.path.join(subject_path, gesture)
                if not os.path.isdir(gesture_path):
                    continue

                for img in os.listdir(gesture_path):
                    if img.lower().endswith((".png", ".jpg", ".jpeg")):
                        self.samples.append(
                            (os.path.join(gesture_path, img), gesture)
                        )

        self.classes = sorted({g for _, g in self.samples})
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.class_to_idx[label]

# ================= TRAINING CONFIG =================
BATCH_SIZE = 32
EPOCHS = 12
LR = 0.001

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = LeapGestureDataset(DATASET_DIR, transform=transform)

# ================= SAVE LABELS (SAME DIRECTORY) =================
labels = {i: g for g, i in dataset.class_to_idx.items()}

print("📍 Writing labels to:", LABELS_PATH)
with open(LABELS_PATH, "w") as f:
    json.dump(labels, f, indent=2)

# ================= SPLIT =================
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# ================= MODEL =================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = GestureNet(num_classes=len(labels)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ================= TRAIN =================
print("🚀 Training started")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    print(f"\n🚀 Epoch {epoch+1}/{EPOCHS}")

    for i, (x, y) in enumerate(train_loader, 1):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 20 == 0:
            print(f"  🔄 Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}")

    # ---------- Validation ----------
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    print(f"📉 Train Loss: {running_loss/len(train_loader):.4f}")
    print(f"🎯 Val Accuracy: {100*correct/total:.2f}%")

# ================= SAVE MODEL =================
torch.save(model.state_dict(), MODEL_PATH)
print("\n✅ Model saved at:", MODEL_PATH)
print("✅ Labels saved at:", LABELS_PATH)
