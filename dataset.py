import numpy as np
import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from config import *

import albumentations as A
from albumentations.pytorch import ToTensorV2


train_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),

    A.HorizontalFlip(p=0.5),

    A.RandomBrightnessContrast(p=0.3),

    A.GaussNoise(p=0.2),

    A.Blur(blur_limit=3, p=0.2),

    A.Normalize(),

    ToTensorV2()
])


test_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),

    A.Normalize(),

    ToTensorV2()
])


class DeepfakeDataset(Dataset):

    def __init__(self, root_dir, transform):

        self.images = []
        self.labels = []
        self.transform = transform

        classes = sorted(
            [d for d in os.listdir(root_dir)
             if os.path.isdir(os.path.join(root_dir, d))]
        )

        self.class_to_idx = {
            cls: idx for idx, cls in enumerate(classes)
        }

        print("Classes:", self.class_to_idx)

        for cls in classes:

            folder = os.path.join(root_dir, cls)

            for img in os.listdir(folder):

                if img.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):

                    self.images.append(os.path.join(folder, img))
                    self.labels.append(self.class_to_idx[cls])

    def __len__(self):

        return len(self.images)

    def __getitem__(self, index):

        image = cv2.imread(self.images[index])

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = self.transform(image=image)["image"]

        label = torch.tensor(self.labels[index], dtype=torch.long)

        return image, label


train_dataset = DeepfakeDataset(
    TRAIN_DIR,
    train_transform
)

test_dataset = DeepfakeDataset(
    TEST_DIR,
    test_transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    persistent_workers=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    persistent_workers=False
)


if __name__ == "__main__":

    print("=" * 50)

    print("Train Images :", len(train_dataset))

    print("Test Images :", len(test_dataset))

    print("=" * 50)

    images, labels = next(iter(train_loader))

    print(images.shape)

    print(labels.shape)