import timm
import torch
import torch.nn as nn

from config import DEVICE


class DeepfakeModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.backbone = timm.create_model(
            "tf_efficientnetv2_s",
            pretrained=True,
            num_classes=0,
            global_pool="avg"
        )

        self.head = nn.Sequential(

            nn.Linear(self.backbone.num_features, 512),

            nn.BatchNorm1d(512),

            nn.ReLU(inplace=True),

            nn.Dropout(0.4),

            nn.Linear(512, 256),

            nn.BatchNorm1d(256),

            nn.ReLU(inplace=True),

            nn.Dropout(0.3),

            nn.Linear(256, 2)

        )

    def forward(self, x):

        x = self.backbone(x)

        x = self.head(x)

        return x


def build_model():

    model = DeepfakeModel().to(DEVICE)

    return model


if __name__ == "__main__":

    model = build_model()

    x = torch.randn(2,3,224,224).to(DEVICE)

    y = model(x)

    print(model)

    print()

    print("Output Shape :",y.shape)

    total = sum(p.numel() for p in model.parameters())

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print()

    print("Total Parameters :",total)

    print("Trainable :",trainable)