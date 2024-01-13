import os
import sys

import torch
from ultralytics import YOLO

import constants
from ai.models.simple_model import SimpleModel
from ai.model_types import ModelType
from ai.models.resnet_model import ResidualModel
from ai.models.yolop_model import YolopModel
from ai.models.cnn_lstm_model import CnnLstmModel
from ai.models.resnet_4ch_model import Residual4ChModel
from constants import MODEL_SELECTION


def load_turning_resnets():
    steering_model = ResidualModel.load_model().cuda()
    left_turn_model = ResidualModel.load_model(constants.LEFT_TURN_MODEL_PATH).cuda()
    no_turn_model = ResidualModel.load_model(constants.NO_TURN_MODEL_PATH).cuda()
    right_turn_model = ResidualModel.load_model(constants.RIGHT_TURN_MODEL_PATH).cuda()
    return steering_model, left_turn_model, no_turn_model, right_turn_model


def load_steering_model():
    if MODEL_SELECTION == ModelType.SIMPLE_MODEL:
        return SimpleModel.load_model()
    elif MODEL_SELECTION == ModelType.RESIDUAL_MODEL:
        return ResidualModel.load_model().cuda()
    elif MODEL_SELECTION == ModelType.YOLOP_MODEL:
        return YolopModel.load_model()
    elif MODEL_SELECTION == ModelType.CNN_LSTM_MODEL:
        return CnnLstmModel.load_model()
    elif MODEL_SELECTION == ModelType.RESIDUAL_4CH_MODEL:
        return Residual4ChModel.load_model()
    else:
        raise Exception("Unknown model type.")


def load_detection_model():
    if constants.DETECTION_MODEL_NAME == 'yolov5s':
        detection_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    elif constants.DETECTION_MODEL_NAME[:6] == 'yolov8':
        detection_model = YOLO(constants.DETECTION_MODEL_NAME + '.pt')  # https://docs.ultralytics.com/modes/
    else:
        raise Exception("Detection model name unknown or not supported.")
    return detection_model

def load_segmentation_model():
    if "cspsg" in constants.SEGMENTATION_MODEL_NAME:
        cspsg_path = os.path.join(constants.ROOT_PATH, "src", "cspsg")
        sys.path.insert(0, cspsg_path)
        segmentation_model = torch.load(constants.SEGMENTATION_MODEL_NAME+'.pt', map_location='cuda')
    elif constants.SEGMENTATION_MODEL_NAME == 'None':
        segmentation_model = None
    else:
        raise Exception("Segmentation model name unknown or not supported.")
    return segmentation_model