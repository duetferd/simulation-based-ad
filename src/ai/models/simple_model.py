import torch.nn as nn

from ai.models.model import Model
from constants import MODEL_PATH


class SimpleModel(Model):
    def __init__(self, transforms=None):
        super().__init__()

        self.transforms = transforms
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(p=0.2)

        self.conv_0 = nn.Conv2d(3, 24, kernel_size=5, stride=2)
        self.conv_1 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.conv_2 = nn.Conv2d(36, 48, kernel_size=5, stride=2)  # original was size 3, is now more like in the papers
        self.conv_3 = nn.Conv2d(48, 64, kernel_size=3)
        self.conv_4 = nn.Conv2d(64, 64, kernel_size=3)

        self.max_pool = nn.MaxPool2d(kernel_size=2)

        self.fc0 = nn.Linear(1024, 256)
        self.fc1 = nn.Linear(256, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        x = x / 127.5 - 1.0
        # Unsqueeze if we only have one image
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = self.elu(self.conv_0(x))
        x = self.elu(self.conv_1(x))
        x = self.dropout(x)
        x = self.max_pool(x)
        x = self.elu(self.conv_2(x))
        x = self.dropout(x)
        x = self.elu(self.conv_3(x))
        x = self.dropout(x)
        x = self.elu(self.conv_4(x))
        x = self.max_pool(x)

        x = x.flatten(start_dim=1)
        x = self.elu(self.fc0(x))
        x = self.dropout(x)
        x = self.elu(self.fc1(x))
        x = self.fc2(x)
        return x

    @staticmethod
    def load_model(model_path=None):
        if model_path is None:
            model_path = MODEL_PATH
        model = SimpleModel().load_from_checkpoint(model_path)
        model.eval()
        return model
