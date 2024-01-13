import os.path

from ai.model_types import ModelType

GERMAN_NUMBERS = False

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# TRAINING_DATA_DIR = os.path.join(ROOT_PATH, "data/6_minutes_drive_random_filtered")
# TRAINING_DATA_DIR = os.path.join(ROOT_PATH, "data/turn_data")
TRAINING_DATA_DIR = os.path.join(ROOT_PATH, "data/75k_balanced_with_extra_data")
# TRAINING_DATA_DIR = os.path.join(ROOT_PATH, "data/training/automatic_driving_58000imgs")

DATAFILE = "VehicleData.txt"

TEST_DATA_DIR = os.path.join(ROOT_PATH, "data", "test_set_only")
MODEL_OUTPUT_DIR = os.path.join(ROOT_PATH, "models")

best_residual = "best/best_residual.ckpt"
best_residual_4ch = "resnet_4ch_model-epoch=11-val_loss=0.01871.ckpt"
best_cnn_lstm = "best/cnn-lstm-model--no_images=3-epoch=10-val_loss=0.04032.ckpt"

# best_residual = "resnet-model-epoch=03-val_loss=0.01273.ckpt" # less steering bias - but actually drives worse

MODEL_SELECTION = ModelType.RESIDUAL_MODEL
if MODEL_SELECTION == ModelType.RESIDUAL_4CH_MODEL:
    TEST_MODEL = os.path.join(MODEL_OUTPUT_DIR, best_residual_4ch)
elif MODEL_SELECTION == ModelType.RESIDUAL_MODEL:
    TEST_MODEL = os.path.join(MODEL_OUTPUT_DIR, best_residual)
elif MODEL_SELECTION == ModelType.CNN_LSTM_MODEL:
    TEST_MODEL = os.path.join(MODEL_OUTPUT_DIR, best_cnn_lstm)
else:
    raise Exception("Model selection not implemented")

INCREMENTAL_TRAINING = False

MODEL_PATH = os.path.join(MODEL_OUTPUT_DIR, TEST_MODEL)
RIGHT_TURN_MODEL_PATH = os.path.join(MODEL_OUTPUT_DIR, "best/right_turn_model.ckpt")
LEFT_TURN_MODEL_PATH = os.path.join(MODEL_OUTPUT_DIR, "best/left_turn_model.ckpt")
NO_TURN_MODEL_PATH = os.path.join(MODEL_OUTPUT_DIR, "best/no_turn_model.ckpt")


RECORDING_PARENT_DIR = os.path.join(ROOT_PATH, "data", "recording")
RECORD_RUN = False

MAX_SPEED = 25
MIN_SPEED = 10

CNN_LSTM_NUMBER_OF_ITEMS = 3

MIN_STEERING_ANGLE = -1
MAX_STEERING_ANGLE = 1
STEERING_BIAS = 0.25  # 0.1 TODO: NEEDS TO BE CHANGED BACK BUT CURRENTLY ALL IMAGES ARE THE SAME
MANUAL_CONTROL = False

# Cropping for steering model in %
CROP_TOP = 0.5
CROP_BOTTOM = 0

# Segmentation model
SEGMENTATION_MODEL_NAME = os.path.join(ROOT_PATH, "models", "cspsg-50epochs-0935acc")

# Cropping for traffic light detection
# in %
CROP_LEFT = 0.3
CROP_RIGHT = 0.4
DETECTION_MODEL_NAME = "yolov8m-seg"  # eg. yolov8n for fast or yolov8x m or l should be fine
DETECTION_PADDING = 0

SHOW_LIVE_DETECTION = True

TRAFFIC_DETECTION_INTERVAL = 0.1
CONTROL_COMPUTATION_INTERVAL = 0.1

# Distance value in range [0, 100] below which values vehicles are considered close
EMERGENCY_BRAKING_DISTANCE_THRESHOLD = 0.55
NO_BREAKING_DISTANCE_THRESHOLD = 0.58

# Ratio between width and height above which another vehicle is ignored by the distance detector
VEHICLE_SIDEWAYS_SHAPE_RATIO = 1.5
CENTER_OFFSET = 0.1  # Offset from the center of the image (0.5) which is considered as the center area of the image

# Changes steering angle normalization and disables flips
TURN_TRAIN_MODE = False
### Clustering elements if one just designs and not really fully trains the model ###
DESIGN_MODE = False
# if DESIGN_MODE:
#     CSV_NAME = "short_driving_log.csv"
#     RECORD_RUN = False

hparams = {
    "TRAINING_DATA_DIR": TRAINING_DATA_DIR,
    "TEST_DATA_DIR": TEST_DATA_DIR,
    "MODEL_OUTPUT_DIR": MODEL_OUTPUT_DIR,
    "MAX_SPEED": MAX_SPEED,
    "MIN_SPEED": MIN_SPEED,
    "MODEL_SELECTION": MODEL_SELECTION,
    "MIN_STEERING_ANGLE": MIN_STEERING_ANGLE,
    "MAX_STEERING_ANGLE": MAX_STEERING_ANGLE,
    "STEERING_BIAS": STEERING_BIAS,
    "CROP_TOP": CROP_TOP,
    "CROP_BOTTOM": CROP_BOTTOM,
    "DESIGN_MODE": DESIGN_MODE
}
