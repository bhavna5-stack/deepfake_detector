import cv2
import torch

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

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    img = transform(image=rgb)["image"]

    img = img.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        output = model(img)

        prob = torch.softmax(output,1)

        conf, idx = torch.max(prob,1)

    label = classes[idx.item()]
    confidence = conf.item()*100

    color = (0,255,0)

    if label == "FAKE":
        color = (0,0,255)

    cv2.putText(
        frame,
        f"{label} {confidence:.2f}%",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.imshow("Deepfake Detector",frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()