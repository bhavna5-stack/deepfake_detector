import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score

from config import *
from dataset import train_loader, test_loader
from model import build_model


model = build_model()

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

scheduler = CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)

scaler = GradScaler()

best_accuracy = 0

start_epoch = 0


if os.path.exists(LAST_CHECKPOINT):

    checkpoint = torch.load(LAST_CHECKPOINT)

    model.load_state_dict(checkpoint["model"])

    optimizer.load_state_dict(checkpoint["optimizer"])

    scheduler.load_state_dict(checkpoint["scheduler"])

    scaler.load_state_dict(checkpoint["scaler"])

    best_accuracy = checkpoint["best_accuracy"]

    start_epoch = checkpoint["epoch"] + 1

    print("Checkpoint Loaded")


def train_one_epoch():
    model.train()

    running_loss = 0
    preds = []
    labels_all = []

    loop = tqdm(train_loader, leave=True)

    for images, labels in loop:

        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast():

            outputs = model(images)

            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        running_loss += loss.item()

        prediction = torch.argmax(outputs, dim=1)

        preds.extend(prediction.detach().cpu().numpy())

        labels_all.extend(labels.detach().cpu().numpy())

        loop.set_postfix(
            loss=loss.item()
        )

    accuracy = accuracy_score(labels_all, preds)

    return running_loss / len(train_loader), accuracy


@torch.no_grad()
def validate():

    model.eval()

    running_loss = 0

    preds = []

    labels_all = []

    loop = tqdm(test_loader, leave=False)

    for images, labels in loop:

        images = images.to(DEVICE, non_blocking=True)

        labels = labels.to(DEVICE, non_blocking=True)

        with autocast():

            outputs = model(images)

            loss = criterion(outputs, labels)

        running_loss += loss.item()

        prediction = torch.argmax(outputs, dim=1)

        preds.extend(prediction.cpu().numpy())

        labels_all.extend(labels.cpu().numpy())

    accuracy = accuracy_score(labels_all, preds)

    return running_loss / len(test_loader), accuracy
if __name__ == "__main__":

    print("=" * 60)
    print("Training Started...")
    print("=" * 60)

    for epoch in range(start_epoch, EPOCHS):

        print(f"\nEpoch [{epoch+1}/{EPOCHS}]")

        train_loss, train_acc = train_one_epoch()

        val_loss, val_acc = validate()

        scheduler.step()

        print(f"Train Loss : {train_loss:.4f}")
        print(f"Train Acc  : {train_acc:.4f}")
        print(f"Val Loss   : {val_loss:.4f}")
        print(f"Val Acc    : {val_acc:.4f}")

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_accuracy": best_accuracy
        }

        torch.save(checkpoint, LAST_CHECKPOINT)

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), BEST_MODEL)
            print("🔥 Best Model Saved")

    print("=" * 60)
    print("Training Finished")
    print(f"Best Validation Accuracy : {best_accuracy:.4f}")
    print("=" * 60)