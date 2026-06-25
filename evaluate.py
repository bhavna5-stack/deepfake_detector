import torch
import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from dataset import test_loader
from model import build_model
from config import DEVICE, BEST_MODEL

model = build_model()
model.load_state_dict(torch.load(BEST_MODEL, map_location=DEVICE))
model.eval()

y_true = []
y_pred = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        preds = torch.argmax(outputs, dim=1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

print("=" * 50)
print("Accuracy :", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall   :", recall_score(y_true, y_pred))
print("F1 Score :", f1_score(y_true, y_pred))
print("=" * 50)

print(classification_report(
    y_true,
    y_pred,
    target_names=["FAKE", "REAL"]
))

print("Confusion Matrix")
print(confusion_matrix(y_true, y_pred))

os.makedirs("outputs", exist_ok=True)

acc = accuracy_score(y_true, y_pred)
pre = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("=" * 60)
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {pre:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1 Score  : {f1:.4f}")
print("=" * 60)

# Classification Report Save
report = classification_report(
    y_true,
    y_pred,
    target_names=["FAKE", "REAL"]
)

with open("outputs/classification_report.txt", "w") as f:
    f.write(report)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["FAKE", "REAL"]
)

fig, ax = plt.subplots(figsize=(6,6))
disp.plot(ax=ax, cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png", dpi=300)
plt.close()

print("Saved:")
print("outputs/classification_report.txt")
print("outputs/confusion_matrix.png")