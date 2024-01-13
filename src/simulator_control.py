import base64
import os
import statistics
import threading
from io import BytesIO
from time import sleep

import eventlet.wsgi
import numpy as np
import socketio
import torch
from PIL import Image

from ai.distance_detection.distance_detector import DistanceDetector
from flask import Flask
from datetime import datetime

from ai.model_types import ModelType
from ai.traffic_light_detection.traffic_light_detector import TrafficLightDetector
from ai.traffic_light_detection.traffic_light_values import TrafficLightValues
from constants import MIN_SPEED, MAX_SPEED, RECORD_RUN, MANUAL_CONTROL, TRAFFIC_DETECTION_INTERVAL, \
    EMERGENCY_BRAKING_DISTANCE_THRESHOLD, NO_BREAKING_DISTANCE_THRESHOLD, CNN_LSTM_NUMBER_OF_ITEMS, MODEL_SELECTION, \
    CONTROL_COMPUTATION_INTERVAL
from control_output import ControlOutput
from data_handling.track_data_element import TrackDataElement

app = Flask(__name__)


class SimulatorControl:
    def __init__(self, steering_model, detection_model, segmentation_model, recording_directory, left_turn_model,
                 no_turn_model, right_turn_model, navigations):
        self.no_turn_model = no_turn_model
        self.right_turn_model = right_turn_model
        self.left_turn_model = left_turn_model
        self.detection_model = detection_model
        self.navigations = navigations
        self.segmentation_model = segmentation_model
        self.steering_model = steering_model.to("cuda")
        self.recording_directory = recording_directory
        self.socket_io = socketio.Server(logger=False, engineio_logger=False)
        self.define_call_backs()
        self.traffic_light_detector = TrafficLightDetector()
        self.distance_detector = DistanceDetector()
        self.middle_image = None
        self.past_steering_angle_values = [0]
        self.past_throttle_values = [0.0, 0.0, 0.0, 0.0]
        self.past_brake_values = [1.0, 1.0, 1.0]
        self.past_images = []
        self.current_controls = None
        self.number_of_intersections = 0
        self.time_last_light = datetime.now()
        self.intersection_mode = False
        self.goal_reached = False
        self.initialized = False

    def run(self):
        print("Start to run")

        wsgi_app = socketio.WSGIApp(self.socket_io, app)
        print("Start to listen")

        threading.Thread(target=self.execute_detectors, daemon=True).start()
        threading.Thread(target=self.execute_drive, daemon=True).start()

        eventlet.wsgi.server(eventlet.listen(('', 4567)), wsgi_app)

    def define_call_backs(self):
        @self.socket_io.on('connect')
        def connect(sid, environment):
            print(f"Connected to sid {sid}.")
            self.emit_empty_data()
            self.traffic_light_detector = TrafficLightDetector()
            self.distance_detector = DistanceDetector()

        @self.socket_io.on('send_image')
        def telemetry(sid, data):
            if data is None:
                print("No data received")
                self.emit_empty_data()
                return
            self.middle_image = self.extract_image(data)

            if len(self.past_images) >= CNN_LSTM_NUMBER_OF_ITEMS:
                self.past_images.pop(0)

            while len(self.past_images) < CNN_LSTM_NUMBER_OF_ITEMS:
                self.past_images.append(self.middle_image)

            if self.current_controls is not None:
                self.emit_control_data()

            if RECORD_RUN:
                self.save_recording()

    @staticmethod
    def extract_image(data):
        # speed = float(data["speed"]) # Speed not available yet TODO Implement if it comes
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        image = np.asarray(image)
        image = image[:, int(image.shape[1] / 3):int(image.shape[1] * 2 / 3),
                :]  # TODO currently using only middle third of the image
        return image

    def drive(self):
        if not self.initialized:
            self.current_controls = ControlOutput(brake=1)
            return

        controls = self.get_controls()
        # controls = self.run_wrong_line_detection(controls)
        controls = self.smooth_controls(controls)
        if abs(controls.steering_angle) < 0.5:
            controls = self.run_distance_detection(controls)

        # print(f"{self.intersection_mode} || {self.traffic_light_detector.traffic_light_detected} {self.traffic_light_detector.previous_color}")
        if not (self.intersection_mode and self.traffic_light_detector.traffic_light_detected == TrafficLightValues.RED
                and self.traffic_light_detector.previous_color == TrafficLightValues.GREEN):
            controls = self.run_traffic_light_detection(controls)

        # controls.steering_angle = min(max(controls.steering_angle, -0.01), 0.01)
        if controls.brake == 1:
            controls.steering_angle = 0

        print(f"{self.traffic_light_detector.traffic_light_detected}")
        self.current_controls = controls

    def run_distance_detection(self, controls):
        distance_to_next_car = self.distance_detector.distance_to_next_vehicle
        if distance_to_next_car <= EMERGENCY_BRAKING_DISTANCE_THRESHOLD:
            controls.brake = 1
            controls.throttle = 0
        elif distance_to_next_car <= NO_BREAKING_DISTANCE_THRESHOLD:
            controls.brake = 0.5
            controls.throttle = 0
        return controls

    def run_traffic_light_detection(self, controls):
        traffic_light_detected = self.traffic_light_detector.traffic_light_detected

        if traffic_light_detected == TrafficLightValues.RED:
            controls.throttle = 0
            controls.brake = 1
            controls.steering_angle = 0

        return controls

    def get_controls(self):
        if MANUAL_CONTROL:
            print("Manual control")
            return ControlOutput.read_manual_controls()

        steering_angle = self.predict_steering_angle()  # Todo: control depending on model
        throttle = self.get_new_throttle(0, steering_angle)
        brake = 1 if throttle == 0 else 0
        if self.goal_reached:
            return ControlOutput(brake=1)
        return ControlOutput(steering_angle=steering_angle, throttle=throttle, brake=brake)

    def predict_steering_angle(self):
        if self.middle_image is None:
            print("No image available")
            return 0

        if MODEL_SELECTION == ModelType.CNN_LSTM_MODEL:
            preprocessed_past_images = [self.get_preprocesses_image_tensor(image, self.segmentation_model) for image in
                                        self.past_images]
            image_tensor = torch.stack(preprocessed_past_images)
        else:
            image_tensor = self.get_preprocesses_image_tensor(self.middle_image, self.segmentation_model)

        command_index = self.number_of_intersections - 1

        if command_index >= len(self.navigations) or (
                command_index == len(self.navigations) - 1 and not self.intersection_mode):
            self.goal_reached = True
            return 0

        if self.intersection_mode:
            if 0 <= command_index < len(self.navigations):
                current_command = self.navigations[command_index]
                if current_command == "left":
                    print(f"LEFT")
                    steering_angle = self.left_turn_model(image_tensor).item()
                elif current_command == "straight":
                    print(f"STRAIGHT")
                    steering_angle = self.no_turn_model(image_tensor).item()
                elif current_command == "right":
                    print(f"RIGHT")
                    steering_angle = self.right_turn_model(image_tensor).item() * 2
                else:
                    raise Exception("Unknown commands.")
            else:
                steering_angle = self.steering_model(image_tensor).item()
        else:
            steering_angle = self.steering_model(image_tensor).item()
        return steering_angle

    @staticmethod
    def get_preprocesses_image_tensor(image, segmentation_model):
        image = TrackDataElement.preprocess(image, segmentation_model=segmentation_model)
        image_tensor = torch.tensor(image, dtype=torch.uint8).to("cuda")
        image_tensor, _ = TrackDataElement.random_grayscale(image_tensor, None, 1)
        image_tensor = image_tensor / 127.5 - 1.0
        return image_tensor

    @staticmethod
    def get_new_throttle(speed, steering_angle):
        if speed < MIN_SPEED:
            throttle = 0.8
        elif speed > MAX_SPEED:
            throttle = 0.0
        else:
            throttle = max(0.8 - abs(steering_angle), 0.2)
        return throttle

    def emit_empty_data(self):
        self.socket_io.emit('manual', data={}, skip_sid=True)

    def emit_control_data(self):
        print(f"Emmitting: {self.current_controls.__dict__()}")
        self.socket_io.emit(
            "control_command",
            data=self.current_controls.__dict__(),
            skip_sid=True)

    def save_recording(self):
        timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
        image_filename = os.path.join(self.recording_directory, timestamp)
        self.middle_image.save(f'{image_filename}.jpg')

    def execute_detectors(self):
        while not self.goal_reached:
            if self.middle_image is not None:
                predictions = self.detection_model.predict(self.middle_image, device=0,
                                                           classes=[0, 1, 2, 3, 5, 7, 9, 11], show=False,
                                                           conf=0.1)
                self.initialized = True
                self.distance_detector.compute_distance_to_next_car(predictions)
                self.traffic_light_detector.detect_traffic_light(predictions, middle_image=self.middle_image)
                self.detect_intersections()

            sleep(TRAFFIC_DETECTION_INTERVAL)

    def detect_intersections(self):
        print(f"INTERSECTION: {self.number_of_intersections}")
        time_since_last_light = datetime.now() - self.time_last_light

        if self.traffic_light_detector.traffic_light_detected != TrafficLightValues.NONE:
            self.time_last_light = datetime.now()
            self.intersection_mode = True

        if time_since_last_light.seconds > 2:
            if self.traffic_light_detector.traffic_light_detected != TrafficLightValues.NONE:
                self.number_of_intersections += 1
            else:
                self.intersection_mode = False
        return self.number_of_intersections

    def execute_drive(self):
        while not self.goal_reached:
            self.drive()
            sleep(CONTROL_COMPUTATION_INTERVAL)

    def smooth_controls(self, controls):
        self.past_throttle_values.append(controls.throttle)
        self.past_throttle_values.pop(0)
        controls.throttle = statistics.fmean(self.past_throttle_values)

        self.past_brake_values.append(controls.brake)
        self.past_brake_values.pop(0)
        controls.brake = max(self.past_brake_values)

        self.past_steering_angle_values.append(controls.steering_angle)
        self.past_steering_angle_values.pop(0)
        controls.steering_angle = min([abs(v) for v in self.past_steering_angle_values]) * np.sign(
            controls.steering_angle)

        return controls
