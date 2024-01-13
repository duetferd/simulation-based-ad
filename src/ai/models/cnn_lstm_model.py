import torch
import torch.nn as nn
import torch.nn.functional as F

from ai.models.model import Model
from ai.models.resnet_model import ResidualModel
from constants import MODEL_PATH, CNN_LSTM_NUMBER_OF_ITEMS


class CnnLstmModel(Model):
    """
        CNN-LSTM architecture taken from: https://github.com/pranoyr/cnn-lstm
    """

    def __init__(self):
        super(CnnLstmModel, self).__init__()
        self.resnet = ResidualModel.load_model().resnet
        # self.resnet.fc = nn.Sequential(nn.Linear(512, 256))
        self.lstm = nn.LSTM(input_size=512, hidden_size=128, num_layers=1)
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, current_and_past_images):
        if len(current_and_past_images.shape) == 4:
            current_and_past_images = current_and_past_images.unsqueeze(0)

        out, hidden = None, None

        for t in range(current_and_past_images.size(1)):
            with torch.no_grad():
                x = self.resnet(current_and_past_images[:, t, :, :, :])
            out, hidden = self.lstm(x.unsqueeze(0), hidden)

        x = self.fc1(out[-1, :, :])
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def training_step(self, train_batch, batch_idx):
        if self.current_epoch == 0:
            # Adding the graph of the model to tensorboard (inputting a random batch to let it run though)
            self.logger.experiment.add_graph(self, train_batch[0])

            # Adding images to be displayed in tensorboard
            image_samples = train_batch[0][:5][:, CNN_LSTM_NUMBER_OF_ITEMS - 1, ...]
            self.logger.experiment.add_image("input", image_samples, self.current_epoch, dataformats="NCHW")
            # we could furthermore add a view into the different layers of the model later

        return super().training_step(train_batch, batch_idx)

    @staticmethod
    def load_model(model_path=None):
        if model_path is None:
            model_path = MODEL_PATH
        model = CnnLstmModel().load_from_checkpoint(model_path, map_location=torch.device("cuda"))
        model.eval()
        return model
