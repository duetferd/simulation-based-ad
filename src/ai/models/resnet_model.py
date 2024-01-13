import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from ai.models.model import Model
from constants import MODEL_PATH


class ResidualModel(Model):
    def __init__(self):
        super().__init__()

        self.upsample = torch.nn.Upsample((256, 256))

        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT).cuda()
        self.resnet.fc = nn.Identity()

        # for param in self.resnet.parameters():
        #     param.requires_grad = False

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = self.upsample(x)
        x = self.resnet(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def training_step(self, train_batch, batch_idx):
        if self.current_epoch == 0:
            # Adding the graph of the model to tensorboard (inputting a random batch to let it run though)
            self.logger.experiment.add_graph(self, train_batch[0])

            # Adding images to be displayed in tensorboard
            self.logger.experiment.add_image("input", train_batch[0][:5], self.current_epoch, dataformats="NCHW")
            # we could furthermore add a view into the different layers of the model later

        return super().training_step(train_batch, batch_idx)

    @staticmethod
    def load_model(model_path=None):
        if model_path is None:
            model_path = MODEL_PATH
        model = ResidualModel().load_from_checkpoint(model_path, map_location=torch.device("cuda"))
        model.eval()
        return model
