import os

import cv2
import numpy as np
import torch
import torchvision

import constants

class TrackDataElement:
    def __init__(self, steering_angle, image_path):
        self.steering_angle = float(steering_angle)
        self.image = self.load_image(image_path)


    @staticmethod
    def load_image(image_path):
        image_path = image_path.replace("\\", "/")
        # image_name = os.path.split(image_path)[1]
        # folder = os.path.split(image_path)[0].split("/")[-2]
        # image_path = os.path.join(data_path, folder, "IMG", image_name)
        # if not os.path.exists(image_path):
        #    raise Exception(f"Image does not exist: '{image_path}'")
        image = cv2.imread(image_path)
        image = np.array(image, dtype=np.uint8)

        # image = torch.tensor(image, dtype=torch.uint8) # would like to put everything on device but does not fit... .to(device)
        return image

    def random_image(self, randnum):
        image = self.image
        steering_angle = self.steering_angle

        if randnum < 0.33:  # Left image
            image, steering_angle = image[:, 0:image.shape[1] // 3,
                                           :], steering_angle + constants.STEERING_BIAS
        elif randnum < 0.66:  # Middle Image
            image, steering_angle = image[:, image.shape[1] // 3:2 * image.shape[1] // 3,
                                           :], steering_angle
        else:  # Right Image
            image, steering_angle = image[:, 2 * image.shape[1] // 3:,
                                           :], steering_angle - constants.STEERING_BIAS
        return image, steering_angle

    @staticmethod
    def preprocess(image_original, disable_preprocessing=False, segmentation_model=None):
        image = image_original
        if not disable_preprocessing:
            image = TrackDataElement.crop_image(image)
            image = TrackDataElement.resize(image)
        image = TrackDataElement.convert_color(image)
        image = TrackDataElement.swap_axes(image)

        if segmentation_model is not None:
            image_segmented = TrackDataElement.predict_segmentation(image_original, segmentation_model)
            image_segmented = image_segmented.cpu().numpy()
            image = np.concatenate((image, image_segmented), axis=0)

        return image

    @staticmethod
    def predict_segmentation(image, segmentation_model):
        image = image.transpose((2, 0, 1))
        image = torch.tensor(image).float()
        image = torchvision.transforms.functional.resize(image, (1024, 2048), antialias=True)

        image = image.to("cuda")
        image = image.unsqueeze(dim=0)
        # # Flip the image horizontally and add it to the batch
        # image_flipped = torchvision.transforms.functional.hflip(image)
        # image = torch.cat((image, image_flipped), dim=0)
        segmentation = segmentation_model(image)[0]
        segmentation = segmentation.argmax(dim=3)
        # Removing dimension 3
        segmentation = segmentation.squeeze(dim=0)
        # add a first dimension to the tensor
        segmentation = segmentation.unsqueeze(dim=0)

        segmentation = segmentation.float()
        segmentation = torchvision.transforms.functional.resize(segmentation, (256, 256), antialias=True)

        return segmentation

    @staticmethod
    def convert_color(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    @staticmethod
    def resize(image):
        image = cv2.resize(image, (256, 256), cv2.INTER_AREA)
        return image

    @staticmethod
    def crop_image(image):
        top_crop_pixel = int(image.shape[0] * constants.CROP_TOP)
        bottom_crop_pixel = int(image.shape[0] * constants.CROP_BOTTOM)

        if bottom_crop_pixel == 0:
            return image[top_crop_pixel:, :, :]

        return image[top_crop_pixel:-bottom_crop_pixel, :, :]

    @staticmethod
    def swap_axes(image):
        return image.transpose((2, 0, 1))

    @staticmethod
    def create_from_row(row, data_path):
        steering_angle = row['steering_angle']
        image_path = os.path.join(data_path, row['image_path'][1:])  # image_path starts with /
        return TrackDataElement(steering_angle, image_path)

    @staticmethod
    def extract_road__by_gray_value(image):
        thresh = 0.4
        image = cv2.GaussianBlur(image, (5, 5), 0)
        gray_scale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray_scale[gray_scale <= 100] = 100
        image = image.astype(np.int32)
        mask = (abs(image[..., 0] - image[..., 1]) + abs(image[..., 0] - image[..., 2]) + abs(
            image[..., 1] - image[..., 2])) <= (thresh * gray_scale)
        image[~mask] = 0
        return image

    @staticmethod
    def extract_road_by_roi(image):
        image = cv2.GaussianBlur(image, (11, 11), 0)
        image = cv2.GaussianBlur(image, (11, 11), 0)
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        roi = gray_image[150:256, 50:206]

        lower_bound = np.percentile(roi.flatten(), 10)
        upper_bound = np.percentile(roi.flatten(), 90)

        mask = cv2.inRange(gray_image, lower_bound, upper_bound)
        result = cv2.bitwise_and(gray_image, gray_image, mask=mask)
        result[result > 0] = 255
        return np.array([result, np.zeros(result.shape), np.zeros(result.shape)]).transpose((1, 2, 0))

    @staticmethod
    def augment(image_tensor, steering_angle):
        if not constants.TURN_TRAIN_MODE:
            image_tensor, steering_angle = TrackDataElement.random_flip(image_tensor, steering_angle, 0.5)
        # image_tensor, steering_angle = TrackDataElement.random_grayscale(image_tensor, steering_angle, 1)
        # image_tensor, steering_angle = TrackDataElement.color_jitter(image_tensor, steering_angle, 0.5)
        # image_tensor, steering_angle = TrackDataElement.random_crop(image_tensor, steering_angle, 0.5)
        image_tensor, steering_angle = TrackDataElement.gaussian_noise(image_tensor, steering_angle, 0.1, 0.25)
        image_tensor, steering_angle = TrackDataElement.random_gaussian_blur(image_tensor, steering_angle, 0.25)
        # image_tensor, steering_angle = TrackDataElement.random_blocks(image_tensor, steering_angle, 0.33)
        return image_tensor, steering_angle

    @staticmethod
    def color_jitter(image_tensor, steering_angle, probability):
        if np.random.rand() < probability:
            image_tensor = torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.1, hue=0.05)(
                image_tensor)
        return image_tensor, steering_angle

    @staticmethod
    def random_flip(image_tensor, steering_angle, probability):
        if np.random.rand() < probability:
            image_tensor = torchvision.transforms.RandomHorizontalFlip(p=1)(image_tensor)
            steering_angle = -steering_angle
        return image_tensor, steering_angle

    # @staticmethod
    # def extract_road(image):
    #     org_image = image
    #     image = cv2.GaussianBlur(image, (25, 25), 0)
    #     # image = cv2.GaussianBlur(image, (25, 25), 0)
    #     gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #     roi = gray_image[500:620, 150:470]
    #     (minVal, maxVal, _, _) = cv2.minMaxLoc(roi)
    #     mask = cv2.inRange(gray_image, minVal, maxVal)
    #     result = cv2.bitwise_and(gray_image, gray_image, mask=mask)
    #     result[result <= 0] = 255
    #     result[result < 255] = 0
    #     result = result[..., np.newaxis]
    #     return np.concatenate([org_image, result], axis=-1)

    @staticmethod
    def random_crop(image_tensor, steering_angle, probability):
        if np.random.rand() < probability:
            image_tensor = torchvision.transforms.RandomCrop((200, 200))(image_tensor)
            image_tensor = torchvision.transforms.Resize((256, 256))(image_tensor)
        return image_tensor, steering_angle

    @staticmethod
    def gaussian_noise(image_tensor, steering_angle, noise_factor, probability):
        if np.random.rand() < probability:
            _min, _max = torch.min(image_tensor), torch.max(image_tensor)
            image_tensor = image_tensor + torch.randn_like(image_tensor) * noise_factor
            image_tensor = torch.clip(image_tensor, _min, _max)
        return image_tensor, steering_angle

    @staticmethod
    def add_noise(image_tensor, noise_factor=0.3):
        noisy = image_tensor + torch.randn_like(image_tensor) * noise_factor
        noisy = torch.clip(noisy, 0., 1.)
        return noisy

    @staticmethod
    def random_blocks(image_tensor, steering_angle, probability):
        image_tensor = torchvision.transforms.RandomErasing(p=probability)(image_tensor)
        image_tensor = torchvision.transforms.RandomErasing(p=probability)(image_tensor)
        return image_tensor, steering_angle

    @staticmethod
    def random_grayscale(image_tensor, steering_angle, probability):
        image_tensor = torchvision.transforms.RandomGrayscale(p=probability)(image_tensor)
        return image_tensor, steering_angle

    @staticmethod
    def random_gaussian_blur(image_tensor, steering_angle, probability):
        if np.random.rand() < probability:
            image_tensor = torchvision.transforms.GaussianBlur(kernel_size=(5, 5))(image_tensor)
        if np.random.rand() < probability:
            image_tensor = torchvision.transforms.GaussianBlur(kernel_size=(5, 5))(image_tensor)
        return image_tensor, steering_angle

    @staticmethod
    def line_detection(image):
        canny_low_threshold = 100
        gray_scale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        canny = cv2.Canny(gray_scale, canny_low_threshold, canny_low_threshold * 3)
        lines = cv2.HoughLinesP(canny, 1, np.pi / 180, 50, np.array([]), 20, 15)
        lines = [] if lines is None else lines
        # result = image
        result = np.zeros(image.shape).astype(np.uint8)
        for x in range(0, len(lines)):
            for x1, y1, x2, y2 in lines[x]:
                if y1 < 150 and y2 < 150:
                    continue

                slope = abs((y2 - y1) / (x2 - x1)) if x2 - x1 != 0 else 1000
                if 0.5 < slope:
                    # cv2.line(inputImage,(x1,y1),(x2,y2),(0,128,0),2, cv2.LINE_AA)
                    pts = np.array([[x1, y1], [x2, y2]], np.int32)
                    cv2.polylines(result, [pts], True, (0, 255, 0), 2)

        return np.array([canny, lines, gray_scale]).transpose((1, 2, 0))
