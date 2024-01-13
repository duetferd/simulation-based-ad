from constants import VEHICLE_SIDEWAYS_SHAPE_RATIO, CENTER_OFFSET

RELEVANT_CLASSES = {0, 1, 2, 3, 5, 7}
MAX_DISTANCE = 1.0


class DistanceDetector:
    def __init__(self):
        self.distance_to_next_vehicle = MAX_DISTANCE
        self.computing = False
        self.initialized = False

    def compute_distance_to_next_car(self, predictions):
        if self.computing:
            return

        masks = predictions[0].masks
        if masks is None:
            self.distance_to_next_vehicle = MAX_DISTANCE
            return

        try:
            self.computing = True
            prediction_data = zip(masks.data.cpu().numpy(), predictions[0].boxes.xyxy.cpu().numpy(),
                                  predictions[0].boxes.cls)
            self.distance_to_next_vehicle = self.get_nearest_vehicle_distance(prediction_data)
            # print(f"Distance to next: {self.distance_to_next_vehicle}")
        finally:
            self.computing = False
            self.initialized = True

    @staticmethod
    def get_nearest_vehicle_distance(predictions_data):
        """
        :return: Value in range [0, MAX_DISTANCE] indicating distance to the closest car in front
            The lower the value, the closer the car.
        """
        closest_vehicle_distance = MAX_DISTANCE
        for mask, box, class_tensor in predictions_data:
            if class_tensor.item() not in RELEVANT_CLASSES:
                continue
            ratio = DistanceDetector.get_vehicle_shape_ratio(box)
            if ratio > VEHICLE_SIDEWAYS_SHAPE_RATIO:
                continue

            x_1, y_1, x_2, y_2 = box
            is_vehicle_in_center = DistanceDetector._is_vehicle_in_center(mask, x_1, x_2, y_1, y_2)
            if not is_vehicle_in_center:
                continue

            last_white_pixel_index = max(y_1, y_2)
            height = mask.shape[0]
            car_distance = MAX_DISTANCE * (1 - (last_white_pixel_index / height))

            closest_vehicle_distance = min(car_distance, closest_vehicle_distance)
        return closest_vehicle_distance

    @staticmethod
    def get_vehicle_shape_ratio(box):
        x_1, y_1, x_2, y_2 = box
        width = abs(x_1 - x_2)
        height = abs(y_1 - y_2)
        ratio = width / height
        return ratio

    @staticmethod
    def _is_vehicle_in_center(mask, x_1, x_2, y_1, y_2):
        image_height = mask.shape[0]
        image_width = mask.shape[1]

        top_location_ratio = y_1 / image_height
        bottom_location_ratio = y_2 / image_height

        left_location_ratio = x_1 / image_width
        right_location_ratio = x_2 / image_width

        image_center = 0.5
        lower_horizontal_threshold = image_center - 0.15
        upper_horizontal_threshold = image_center + 0.15

        top_vertical_threshold = 0.35
        bottom_vertical_threshold = 0.9

        is_horizontally_in_center = (lower_horizontal_threshold <= left_location_ratio <= upper_horizontal_threshold) or \
                                    (lower_horizontal_threshold <= right_location_ratio <= upper_horizontal_threshold)

        is_vertically_in_center = (top_vertical_threshold <= top_location_ratio <= bottom_vertical_threshold) or \
                                  (top_vertical_threshold <= bottom_location_ratio <= bottom_vertical_threshold)

        return is_vertically_in_center and is_horizontally_in_center
