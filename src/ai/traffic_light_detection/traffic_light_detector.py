from datetime import datetime
from time import sleep

import numpy as np

import cv2

from ai.traffic_light_detection.traffic_light_values import TrafficLightValues
from constants import SHOW_LIVE_DETECTION, CROP_LEFT, \
    CROP_RIGHT, DETECTION_MODEL_NAME, DETECTION_PADDING


class TrafficLightDetector:
    def __init__(self):
        self.traffic_light_detected = TrafficLightValues.NONE
        self.computing = False
        self.initialized = False
        self.previous_color = TrafficLightValues.NONE
        self.last_previous_color_update = datetime.now().second
        self.history = [TrafficLightValues.NONE, TrafficLightValues.NONE, TrafficLightValues.NONE]

    def detect_traffic_light(self, predictions, middle_image):
        if self.computing:
            return

        try:
            self.computing = True

            traffic_lights = self.crop_out_traffic_lights(predictions, middle_image)
            filtered_images = self.filter_middle_images(traffic_lights, middle_image.shape, CROP_LEFT, CROP_RIGHT)

            max_color = TrafficLightValues.NONE
            max_pixels = 0

            for image in filtered_images:
                color, number_of_pixels = self.find_primary_color(image["image"])
                if color is None:
                    continue

                if number_of_pixels > max_pixels:
                    max_color = color
                    max_pixels = number_of_pixels

            if max_pixels < 4:
                traffic_light_detected = TrafficLightValues.NONE
            else:
                new_value = TrafficLightValues(max_color)
                if self.previous_color == TrafficLightValues.GREEN and new_value == TrafficLightValues.RED:
                    sleep(1)

                self.previous_color = self.traffic_light_detected
                self.last_previous_color_update = datetime.now().second
                traffic_light_detected = new_value

            self.history.pop(0)
            self.history.append(traffic_light_detected)
            only_color_history = [x for x in self.history if x != TrafficLightValues.NONE]
            if only_color_history:
                self.traffic_light_detected = max(only_color_history, key=only_color_history.count)
            else:
                self.traffic_light_detected = TrafficLightValues.NONE

        finally:
            # print(f"TRAFFIC_LIGHT_DETECTED: {self.traffic_light_detected}")
            if datetime.now().second - self.last_previous_color_update > 1:
                self.previous_color = TrafficLightValues.NONE

            self.computing = False
            self.initialized = True

    def find_primary_color(self, image):
        # convert image to HSV
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # define range of red color in HSV
        lower_red = np.array([157, 61, 102])
        upper_red = np.array([180, 255, 255])

        # define range of green color in HSV
        lower_green = np.array([50, 50, 50])
        upper_green = np.array([85, 255, 255])

        # define range of yellow color in HSV
        lower_yellow = np.array([22, 139, 98])
        upper_yellow = np.array([41, 255, 255])

        # Threshold the HSV image to get only red colors
        mask_red = cv2.inRange(hsv, lower_red, upper_red)

        # Threshold the HSV image to get only green colors
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        # Threshold the HSV image to get only yellow colors
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # count the number of red and green pixels
        red_pixels = cv2.countNonZero(mask_red)
        green_pixels = cv2.countNonZero(mask_green)
        yellow_pixels = cv2.countNonZero(mask_yellow)

        # return the dominant color
        if red_pixels > green_pixels and red_pixels > yellow_pixels:
            return 'red', red_pixels
        elif green_pixels > red_pixels and green_pixels > yellow_pixels:
            return 'green', green_pixels
        elif yellow_pixels > red_pixels and yellow_pixels > green_pixels:
            return 'yellow', yellow_pixels
        else:
            return None, 0

    # a function to detect the color of the traffic light
    def crop_out_traffic_lights(self, predictions, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original = image.copy()

        # show the results at every 10th frame
        if SHOW_LIVE_DETECTION:
            # rendered = results.render()[0]
            rendered = predictions[0].plot()
            width = rendered.shape[1]
            cropOff = int(width * CROP_LEFT)
            # rendered = rendered[:, cropOff:width-cropOff, :]
            rendered = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)
            self.show_live_stream(rendered)

        # crop the image to create multiple images of traffic lights
        traffic_lights = []
        if DETECTION_MODEL_NAME[:6] == "yolov5":
            boxes = predictions.xyxy[0]
        elif DETECTION_MODEL_NAME[:6] == "yolov8":
            boxes = predictions[0].boxes
        else:
            print("ERROR: invalid detection model name")

        for i in range(len(boxes)):
            if DETECTION_MODEL_NAME[:6] == "yolov5":
                xmin, ymin, xmax, ymax, probability, class_id = boxes.xyxy[i]
            elif DETECTION_MODEL_NAME[:6] == "yolov8":
                xmin, ymin, xmax, ymax = boxes.xyxy[i]
                class_id = boxes.cls[i]
            else:
                print("ERROR: invalid detection model name")
            if class_id == 9:
                image = original[int(ymin) - DETECTION_PADDING:int(ymax) + DETECTION_PADDING,
                        int(xmin) - DETECTION_PADDING:int(xmax) + DETECTION_PADDING]
                # show the image
                traffic_lights.append(
                    {"image": image, "xmin": xmin, "ymin": ymin, "height": xmax - xmin, "width": ymax - ymin})
        return traffic_lights

    def filter_middle_images(self, images, image_shape, crop_left, crop_right):
        # only keep the images that are about the middle of image
        filtered_images = []
        width = image_shape[1]
        for image in images:
            if image["xmin"] > width * crop_left and image["xmin"] < width * (1 - crop_right):
                filtered_images.append(image)
        return filtered_images

    def show_live_stream(self, image):
        cv2.imshow('live_stream', image)
        cv2.waitKey(1)

    def normalize_images(self, traffic_lights):
        for tl_dict in traffic_lights.values():
            tl_dict["image"] = cv2.resize(tl_dict["image"], (10, 10))
        return traffic_lights
