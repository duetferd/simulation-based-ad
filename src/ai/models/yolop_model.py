import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from ai.models.model import Model
from constants import MODEL_PATH


class YolopModel(Model):
    def __init__(self):
        super().__init__()

        self.upsample = torch.nn.Upsample((256, 256))

        self.yolop = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)

        for param in self.yolop.parameters():
            param.requires_grad = False

        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = self.upsample(x)
        _, x, _ = self.yolop(x)
        x = self.resnet(x[:, 0, ...].unsqueeze(1).repeat((1, 3, 1, 1)))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    @staticmethod
    def load_model(model_path=None):
        if model_path is None:
            model_path = MODEL_PATH
        model = YolopModel().load_from_checkpoint(model_path)
        model.eval()
        return model
