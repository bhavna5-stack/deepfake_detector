from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime
import pandas as pd
import os

styles = getSampleStyleSheet()

os.makedirs("outputs", exist_ok=True)

pdf = SimpleDocTemplate("outputs/report.pdf")

story = []

story.append(Paragraph("<b>Deepfake Detection Report</b>", styles["Title"]))
story.append(Spacer(1,20))

story.append(Paragraph(
    f"Generated : {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}",
    styles["Normal"]
))

story.append(Spacer(1,20))

if os.path.exists("outputs/history.csv"):

    df = pd.read_csv("outputs/history.csv")

    last = df.iloc[-1]

    story.append(Paragraph(
        f"<b>Prediction :</b> {last['Prediction']}",
        styles["Normal"]
    ))

    story.append(Paragraph(
        f"<b>Confidence :</b> {last['Confidence']} %",
        styles["Normal"]
    ))

    story.append(Paragraph(
        f"<b>Fake Probability :</b> {last['Fake']} %",
        styles["Normal"]
    ))

    story.append(Paragraph(
        f"<b>Real Probability :</b> {last['Real']} %",
        styles["Normal"]
    ))

else:

    story.append(
        Paragraph(
            "No Prediction History Found.",
            styles["Normal"]
        )
    )

story.append(Spacer(1,20))

story.append(
    Paragraph(
        "<b>Model :</b> EfficientNetV2-S",
        styles["Normal"]
    )
)

story.append(
    Paragraph(
        "<b>Framework :</b> PyTorch",
        styles["Normal"]
    )
)

story.append(
    Paragraph(
        "<b>Device :</b> CUDA",
        styles["Normal"]
    )
)

story.append(
    Paragraph(
        "<b>Author :</b> Tanuj Kumar Singh",
        styles["Normal"]
    )
)

pdf.build(story)

print("PDF Saved -> outputs/report.pdf")