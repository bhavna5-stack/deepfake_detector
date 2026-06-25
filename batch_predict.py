import os
import cv2
import torch
import pandas as pd

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

classes = ["FAKE", "REAL"]


def predict(img_path):

    img = cv2.imread(img_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = transform(image=img)["image"]

    img = img.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        out = model(img)

        prob = torch.softmax(out,1)

        conf, idx = torch.max(prob,1)

    return (
        classes[idx.item()],
        round(conf.item()*100,2)
    )


if __name__ == "__main__":

    folder = input("Folder Path : ")

    rows = []

    files = os.listdir(folder)[:20

    if file.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp")):

        path = os.path.join(folder,file)

        pred,conf = predict(path)

        rows.append([file,pred,conf])

df = pd.DataFrame(
    rows,
    columns=[
        "Image",
        "Prediction",
        "Confidence"
    ]
)

df.to_csv("outputs/batch_resfolder = inpuults.csv",index=False)

print(df)

print("\nResults Saved -> outputs/batch_results.csv")

