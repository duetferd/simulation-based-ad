import os

import yaml

from constants import RECORD_RUN, RECORDING_PARENT_DIR, ROOT_PATH


def prepare_recording_and_get_directory():
    current_recording_directory = None
    if RECORD_RUN:
        current_recording_directory = setup_recording_directories()
        print(f"This run will be saved at '{current_recording_directory}'.")
    else:
        print("This run will not be recorded.")
    return current_recording_directory


def setup_recording_directories():
    if not os.path.exists(RECORDING_PARENT_DIR):
        os.makedirs(RECORDING_PARENT_DIR)

    recording_directory = get_recording_directory()
    if not os.path.exists(recording_directory):
        os.makedirs(recording_directory)
    return recording_directory


def get_recording_directory():
    number_of_files = len(os.listdir(RECORDING_PARENT_DIR))
    return os.path.join(RECORDING_PARENT_DIR, str(number_of_files))


def read_navigations():
    yaml_path = os.path.join(ROOT_PATH, "navigations.yaml")
    try:
        with open(yaml_path, "r") as f:
            navigations = yaml.safe_load(f)["navigations"]
    except:
        navigations = ["right", "left", "straight", "left", "right"]
        print(f"Failed to load yaml navigations file. Using default navigations instead: {navigations}.")
    return navigations
