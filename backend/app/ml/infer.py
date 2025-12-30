import json
import torch
from torchvision import transforms
from PIL import Image
import os

from app.ml.model import GestureNet

# ================= PATHS =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "gesture_model.pth")
LABELS_PATH = os.path.join(BASE_DIR, "labels.json")

# ================= LABEL NORMALIZATION =================
def normalize_label(label: str) -> str:
    """
    Converts:
    '01_palm'        -> 'palm'
    '02_l'           -> 'l'
    '04_fist_moved'  -> 'fist_moved'
    '08_palm_moved'  -> 'palm_moved'
    """
    # Remove numeric prefix only (before first underscore)
    if "_" in label:
        return label.split("_", 1)[1]
    return label

# ================= LOAD LABELS =================
with open(LABELS_PATH, "r") as f:
    raw_labels = json.load(f)

idx_to_label = {
    int(k): normalize_label(v)
    for k, v in raw_labels.items()
}

# ================= LOAD MODEL =================
device = "cuda" if torch.cuda.is_available() else "cpu"

model = GestureNet(num_classes=len(idx_to_label))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ================= IMAGE TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ================= INFERENCE =================
def predict_gesture(image_path: str):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)

    return {
        "gesture": idx_to_label[pred.item()],
        "confidence": round(conf.item(), 4)
    }
