import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset

import constants
from ai.ai_utils import load_segmentation_model
from ai.model_types import ModelType
from data_handling.track_data_element import TrackDataElement

from constants import MIN_STEERING_ANGLE, MAX_STEERING_ANGLE, DATAFILE, CNN_LSTM_NUMBER_OF_ITEMS


class TrackDataset(Dataset):
    def __init__(self, data_path, transforms=None, dataset_limit=None, disable_data_augmentation=False, lstm_mode=False,
                 disable_preprocessing=False):
        self.disable_preprocessing = disable_preprocessing
        self.lstm_mode = lstm_mode
        self.disable_data_augmentation = disable_data_augmentation
        self.data = TrackDataset.read_data(data_path, dataset_limit)
        self.transforms = transforms
        self.val_ranges = []
        self.segmentation_model = load_segmentation_model() if constants.MODEL_SELECTION == ModelType.RESIDUAL_4CH_MODEL else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item_index):
        random_number = np.random.rand()
        current_image, steering_angle = self.get_element(item_index, random_number)
        if not self.lstm_mode:
            return current_image, steering_angle

        past_images = [self.get_element(max(0, i), random_number)[0] for i in
                       range(item_index - 1, item_index - CNN_LSTM_NUMBER_OF_ITEMS, -1)]

        if len(past_images) <= 0:
            return current_image.unsqueeze(0), steering_angle

        past_images = torch.stack(past_images)
        past_and_current_images = torch.cat((past_images, current_image.unsqueeze(0)))
        return past_and_current_images, steering_angle

    def get_element(self, i, randnum):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data_element = self.data[i]
        image, steering_angle = self.get_image_and_steering_angle(data_element, randnum)
        image = TrackDataElement.preprocess(image, self.disable_preprocessing, self.segmentation_model)
        image_tensor = torch.tensor(image, dtype=torch.uint8)
        image_tensor = image_tensor / 127.5 - 1.0
        image_tensor, _ = TrackDataElement.random_grayscale(image_tensor, None, 1)
        if self.should_transform(i):
            image_tensor, steering_angle = self.apply_transforms(image_tensor, steering_angle)
        return image_tensor.to(device), torch.tensor(steering_angle).to(device)

    def apply_transforms(self, image_tensor, steering_angle):
        if self.transforms is not None and not self.disable_data_augmentation:
            for transform in self.transforms:
                image_tensor, steering_angle = transform(image_tensor, steering_angle)
        return image_tensor, steering_angle

    def split_train_val_data(self):
        start_point = 0
        train_sets = []
        val_sets = []

        for i in range(0, 5):
            start_point = self.create_subset_and_get_new_index(start_point, train_sets, 0.16)
            start_point = self.create_subset_and_get_new_index(start_point, val_sets, 0.04, self.val_ranges)

        train_dataset = torch.utils.data.ConcatDataset(train_sets)
        val_dataset = torch.utils.data.ConcatDataset(val_sets)
        return train_dataset, val_dataset

    def create_subset_and_get_new_index(self, start_point, datasets, current_percentage, ranges=None):
        end_point = start_point + int(len(self) * current_percentage)
        current_range = range(start_point, end_point)
        if ranges is not None:
            ranges.append(current_range)
        datasets.append(Subset(self, current_range))
        return end_point

    @staticmethod
    def get_image_and_steering_angle(data_element, randnum):
        image, steering_angle = data_element.random_image(randnum)
        steering_angle = max(MIN_STEERING_ANGLE, min(MAX_STEERING_ANGLE, steering_angle))
        return image, steering_angle

    @staticmethod
    def read_data(data_path, limit):
        data = []
        if DATAFILE == "driving_log.csv":
            vehicle_data_path = os.path.join(data_path, DATAFILE)
            df = pd.read_csv(vehicle_data_path, sep=",", nrows=limit)
        else:
            vehicle_data_path = os.path.join(data_path, "VehicleData.txt")
            df = pd.read_csv(vehicle_data_path, sep=" ", header=None, nrows=limit,
                             names=["image_path", "throttle", "break", "steering_angle", "velocity"])

        steering_angle_column = df["steering_angle"]
        min_value = steering_angle_column.min() if not constants.TURN_TRAIN_MODE else -30
        max_value = steering_angle_column.max() if not constants.TURN_TRAIN_MODE else 30

        df["steering_angle"] = ((steering_angle_column - min_value) / (
                max_value - min_value)) * 2 - 1

        for _, row in df.iterrows():
            try:
                new_track_data_element = TrackDataElement.create_from_row(row, data_path)
                data.append(new_track_data_element)
            except Exception:
                pass
        return data

    def should_transform(self, i):
        for current_range in self.val_ranges:
            if i in current_range:
                return False
        return True
