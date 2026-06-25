import os
import cv2
import torch
import pandas as pd
from datetime import datetime
import importlib

# -------------------------------
# Albumentations
# -------------------------------
try:
    A = importlib.import_module("albumentations")

    try:
        alb_pytorch = importlib.import_module("albumentations.pytorch")
        ToTensorV2 = getattr(alb_pytorch, "ToTensorV2")
    except:
        alb_pytorch = importlib.import_module("albumentations.pytorch.transforms")
        ToTensorV2 = getattr(alb_pytorch, "ToTensorV2")

    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(),
        ToTensorV2()
    ])

except:

    from PIL import Image
    import torchvision.transforms as T

    tf = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(
            (0.485,0.456,0.406),
            (0.229,0.224,0.225)
        )
    ])

    def transform(image=None):
        image = Image.fromarray(image)
        return {"image": tf(image)}

# -------------------------------
# Model
# -------------------------------

from model import build_model
from config import DEVICE, BEST_MODEL

model = build_model()

model.load_state_dict(
    torch.load(
        BEST_MODEL,
        map_location=DEVICE
    )
)

model.eval()

classes = ["FAKE","REAL"]

# -------------------------------
# Face Detector
# -------------------------------

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# -------------------------------
# Prediction
# -------------------------------

def predict(image_path):

    image = cv2.imread(image_path)

    if image is None:
        print("\nImage not found.")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(60,60)
    )

    # Face crop
    if len(faces) > 0:

        x,y,w,h = faces[0]

        image = image[y:y+h, x:x+w]

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = transform(image=image)["image"]

    image = image.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        output = model(image)

        prob = torch.softmax(output,1)

        confidence,index = torch.max(prob,1)

    fake_prob = prob[0][0].item()*100
    real_prob = prob[0][1].item()*100

    print("="*60)
    print("Prediction :",classes[index.item()])
    print("Confidence :",round(confidence.item()*100,2),"%")
    print("Fake Probability :",round(fake_prob,2),"%")
    print("Real Probability :",round(real_prob,2),"%")
    print("="*60)

    # -------------------------------
    # Save History
    # -------------------------------

    history = "outputs/history.csv"

    os.makedirs("outputs",exist_ok=True)

    row = pd.DataFrame([{
        "Time":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Prediction":classes[index.item()],
        "Confidence":round(confidence.item()*100,2),
        "Fake_Probability":round(fake_prob,2),
        "Real_Probability":round(real_prob,2)
    }])

    if os.path.exists(history):

      row.to_csv(
    history,
    mode="a",
    header=False,
    index=False,
    lineterminator="\n"
)
    else:

        row.to_csv(
            history,
            index=False
        )

    print("Prediction Saved -> outputs/history.csv")


# -------------------------------
# Main
# -------------------------------

if __name__ == "__main__":

    path = input("\nImage Path : ")

    predict(path) 