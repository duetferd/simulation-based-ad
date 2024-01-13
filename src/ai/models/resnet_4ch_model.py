import torch
import torch.nn as nn
from torch import optim
from torchvision.models import resnet18, ResNet18_Weights

from ai.models.model import Model
from constants import MODEL_PATH


class Residual4ChModel(Model):
    def __init__(self):
        super().__init__()

        self.upsample = torch.nn.Upsample((256, 256))

        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT).cuda()
        self.resnet.fc = nn.Identity()

        self.resnet2 = resnet18(weights=ResNet18_Weights.DEFAULT).cuda()
        self.resnet2.fc = nn.Identity()

        # for param in self.resnet.parameters():
        #     param.requires_grad = False

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = self.upsample(x)
        x1 = self.resnet(x[:, :3])
        segments = x[:, 3]
        # copying the segments to the 3 channels
        segments = segments.unsqueeze(1)
        segments = segments.repeat(1, 3, 1, 1)
        x2 = self.resnet2(segments)
        x = torch.cat((x1, x2), dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            }
        }

    def training_step(self, train_batch, batch_idx):
        if self.current_epoch == 0:
            # Adding the graph of the model to tensorboard (inputting a random batch to let it run though)
            self.logger.experiment.add_graph(self, train_batch[0])
            
            # Adding images to be displayed in tensorboard
            #self.logger.experiment.add_image("input", train_batch[0][:5][:,:3], self.current_epoch, dataformats="NCHW")
            # we could furthermore add a view into the different layers of the model later

        x, y = train_batch
        y_hat = self.forward(x)
        y = y.squeeze()
        y_hat = y_hat.squeeze()
        assert y_hat.shape == y.shape
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_hat = self.forward(x)
        y = y.squeeze()
        y_hat = y_hat.squeeze()
        assert y_hat.shape == y.shape
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y_hat = self.forward(x)
        y = y.squeeze()
        y_hat = y_hat.squeeze()
        assert y_hat.shape == y.shape
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    @staticmethod
    def load_model(model_path=None):
        if model_path is None:
            model_path = MODEL_PATH
        model = Residual4ChModel().load_from_checkpoint(model_path, map_location=torch.device("cuda"))
        model.eval()
        return model
