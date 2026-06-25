import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import IMAGE_SIZE


train_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),

    A.HorizontalFlip(p=0.5),

    A.VerticalFlip(p=0.2),

    A.RandomRotate90(p=0.3),

    A.ShiftScaleRotate(
        shift_limit=0.05,
        scale_limit=0.10,
        rotate_limit=20,
        p=0.5
    ),

    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.5
    ),

    A.GaussNoise(
        p=0.2
    ),

    A.Blur(
        blur_limit=3,
        p=0.2
    ),

    A.CoarseDropout(
        max_holes=8,
        max_height=24,
        max_width=24,
        p=0.3
    ),

    A.Normalize(),

    ToTensorV2()
])


valid_transform = A.Compose([

    A.Resize(IMAGE_SIZE, IMAGE_SIZE),

    A.Normalize(),

    ToTensorV2()

])