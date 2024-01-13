import constants
from ai.ai_utils import load_detection_model, load_segmentation_model, load_turning_resnets
from ai.model_types import ModelType
from simulator_control import SimulatorControl
from data_handling_utils import prepare_recording_and_get_directory, read_navigations

print("Make sure you insert your own IP address in the test simulator Data folder! (IP.txt)")

if __name__ == '__main__':
    recording_directory = prepare_recording_and_get_directory()
    steering_model, left_turn_model, no_turn_model, right_turn_model = load_turning_resnets()
    detection_model = load_detection_model()
    segmentation_model = load_segmentation_model() if constants.TEST_MODEL == ModelType.RESIDUAL_4CH_MODEL else None
    navigations = read_navigations()

    simulator_control = SimulatorControl(steering_model, detection_model, segmentation_model, recording_directory,
                                         left_turn_model, no_turn_model, right_turn_model, navigations)
    simulator_control.run()
