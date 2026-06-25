import gradio as gr
import torch
import cv2
import pandas as pd
from datetime import datetime
import os

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(),
        ToTensorV2()
    ])
except:
    from PIL import Image
    import torchvision.transforms as T

    _transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize((0.485,0.456,0.406),
                    (0.229,0.224,0.225))
    ])

    def transform(image=None):
        image = Image.fromarray(image)
        return {"image": _transform(image)}

from model import build_model
from config import DEVICE, BEST_MODEL

model = build_model()
model.load_state_dict(torch.load(BEST_MODEL, map_location=DEVICE))
model.eval()

classes = ["FAKE", "REAL"]


def predict(image):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = transform(image=image)["image"]

    image = image.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        output = model(image)

        prob = torch.softmax(output, dim=1)

        confidence, index = torch.max(prob, dim=1)

    fake = round(prob[0][0].item()*100,2)
    real = round(prob[0][1].item()*100,2)

    history = "outputs/history.csv"

    row = pd.DataFrame([{
        "Time":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Prediction":classes[index.item()],
        "Confidence":round(confidence.item()*100,2),
        "Fake":fake,
        "Real":real
    }])

    if os.path.exists(history):
        row.to_csv(history,mode="a",header=False,index=False)
    else:
        row.to_csv(history,index=False)

    return (
        f"{classes[index.item()]}",
        f"{confidence.item()*100:.2f} %",
        {"FAKE":fake,"REAL":real}
    )


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Confidence"),
        gr.Label(label="Probability")
    ],
    title="Deepfake Detector",
    description="Upload an image"
)

demo.launch()