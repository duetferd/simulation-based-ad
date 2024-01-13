from enum import Enum


class ModelType(Enum):
    SIMPLE_MODEL = "simple_model"
    RESIDUAL_MODEL = "residual_model"
    YOLOP_MODEL = "yolop_model"
    RESIDUAL_4CH_MODEL = "residual_4ch_model"
    CNN_LSTM_MODEL = "cnn_lstm_model"

