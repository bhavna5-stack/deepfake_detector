import cv2
import torch
import os
from tqdm import tqdm

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    transform = A.Compose([
        A.Resize(224,224),
        A.Normalize(),
        ToTensorV2()
    ])
except:
    from PIL import Image
    import torchvision.transforms as T

    tf = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize((0.485,0.456,0.406),
                    (0.229,0.224,0.225))
    ])

    def transform(image=None):
        image = Image.fromarray(image)
        return {"image": tf(image)}

from model import build_model
from config import DEVICE, BEST_MODEL

model = build_model()
model.load_state_dict(torch.load(BEST_MODEL, map_location=DEVICE))
model.eval()

classes = ["FAKE","REAL"]


def predict_frame(frame):

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame = transform(image=frame)["image"]

    frame = frame.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        out = model(frame)

        prob = torch.softmax(out,1)

        conf, idx = torch.max(prob,1)

    return classes[idx.item()], conf.item()*100


video = input("Video Path : ")

cap = cv2.VideoCapture(video)

total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fake = 0
real = 0

for _ in tqdm(range(total)):

    ret, frame = cap.read()

    if not ret:
        break

    pred, conf = predict_frame(frame)

    if pred == "FAKE":
        fake += 1
    else:
        real += 1

cap.release()

print("="*50)
print("Fake Frames :", fake)
print("Real Frames :", real)

if fake > real:
    print("\nFinal Prediction : FAKE VIDEO")
else:
    print("\nFinal Prediction : REAL VIDEO")