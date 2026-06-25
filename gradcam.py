import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from model import build_model
from config import DEVICE, BEST_MODEL

model = build_model()
model.load_state_dict(torch.load(BEST_MODEL, map_location=DEVICE))
model.eval()

target_layer = model.backbone.conv_head

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.485,0.456,0.406),
        (0.229,0.224,0.225)
    )
])

image_path = input("Image Path : ")

rgb = Image.open(image_path).convert("RGB")

rgb = rgb.resize((224,224))

rgb_np = np.array(rgb).astype(np.float32) / 255.0

tensor = transform(rgb).unsqueeze(0).to(DEVICE)

cam = GradCAM(
    model=model,
    target_layers=[target_layer]
)

grayscale_cam = cam(
    input_tensor=tensor
)[0]

import cv2

grayscale_cam = cv2.resize(
    grayscale_cam,
    (224,224)
)

visualization = show_cam_on_image(
    rgb_np,
    grayscale_cam,
    use_rgb=True
)


cv2.imwrite(
    "outputs/gradcam.jpg",
    cv2.cvtColor(
        visualization,
        cv2.COLOR_RGB2BGR
    )
)

print("Saved -> outputs/gradcam.jpg")