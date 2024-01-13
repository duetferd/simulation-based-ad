import yaml

from constants import GERMAN_NUMBERS


class ControlOutput:
    def __init__(self, steering_angle=0, throttle=0, brake=0):
        self.steering_angle = steering_angle
        self.throttle = throttle
        self.brake = brake

    def __dict__(self) -> dict:
        if GERMAN_NUMBERS:
            return {
                "steering_angle": str(self.steering_angle).replace(".", ","),
                "throttle": str(self.throttle).replace(".", ","),
                "brake": str(self.brake).replace(".", ",")
            }

        return {
            "steering_angle": str(self.steering_angle),
            "throttle": str(self.throttle),
            "brake": str(self.brake)
        }

    def __str__(self) -> str:
        return f"Emitting data: steering_angle: {self.steering_angle}, throttle: {self.throttle}, brake: {self.brake}"

    @staticmethod
    def read_manual_controls():
        with open("controls.yaml", "r") as stream:
            controls = yaml.safe_load(stream)
        return ControlOutput.from_dict(controls)

    @staticmethod
    def from_dict(control_dict):
        return ControlOutput(
            control_dict.get("steering_angle", 0),
            control_dict.get("throttle", 0),
            control_dict.get("brake", 0)
        )
