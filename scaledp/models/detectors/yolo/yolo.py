from typing import Any, Tuple

import logging
import cv2
import numpy as np
import onnxruntime

from scaledp.enums import Device
from scaledp.models.detectors.yolo.utils import multiclass_nms, xywh2xyxy


class YOLO:

    def __init__(self, path, device=Device.CPU, conf_thres=0.7, iou_thres=0.5) -> None:
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres

        # Store original image dimensions and scaling info
        self.original_width = None
        self.original_height = None
        self.scale_factor = None
        self.pad_x = None
        self.pad_y = None

        # Initialize model
        self.initialize_model(path, device)

    def __call__(self, image) -> Any:
        return self.detect_objects(image)

    def initialize_model(self, path, device):
        provider = (
            "CUDAExecutionProvider" if device == Device.CUDA else "CPUExecutionProvider"
        )

        if provider in onnxruntime.get_available_providers():
            providers = [provider]
        else:
            logging.warning(
                f"{provider} is not available. Falling back to CPUExecutionProvider."
            )
            providers = ["CPUExecutionProvider"]
        self.session = onnxruntime.InferenceSession(path, providers=providers)
        # Get model info
        self.get_input_details()
        self.get_output_details()

    def detect_objects(self, image):
        input_tensor = self.prepare_input(image)

        # Perform inference on the image
        outputs = self.inference(input_tensor)

        self.boxes, self.scores, self.class_ids = self.process_output(outputs)

        return self.boxes, self.scores, self.class_ids

    def rescale_image_with_padding(
        self, image: np.ndarray, target_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Rescale image while keeping aspect ratio and pad with white background.

        Args:
            image: Input image (H, W, C)
            target_size: Target size (width, height)

        Returns:
            Rescaled and padded image
        """
        self.original_height, self.original_width = image.shape[:2]
        target_width, target_height = target_size

        # Calculate scaling factor to maintain aspect ratio
        scale_w = target_width / self.original_width
        scale_h = target_height / self.original_height
        self.scale_factor = min(scale_w, scale_h)

        # Calculate new dimensions
        new_width = int(self.original_width * self.scale_factor)
        new_height = int(self.original_height * self.scale_factor)

        # Resize image
        resized_image = cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_LINEAR
        )

        # Calculate padding to center the image
        self.pad_x = (target_width - new_width) // 2
        self.pad_y = (target_height - new_height) // 2

        # Create padded image with white background
        padded_image = np.full((target_height, target_width, 3), 255, dtype=np.uint8)

        # Calculate the actual placement bounds to avoid index errors
        end_y = min(self.pad_y + new_height, target_height)
        end_x = min(self.pad_x + new_width, target_width)

        # Adjust the resized image if it exceeds target bounds
        actual_height = end_y - self.pad_y
        actual_width = end_x - self.pad_x

        # Place the resized image in the center of the padded image
        padded_image[self.pad_y : end_y, self.pad_x : end_x] = resized_image[
            :actual_height, :actual_width
        ]

        return padded_image

    def restore_coordinates(self, boxes: np.ndarray) -> np.ndarray:
        """
        Restore bounding box coordinates to original image space.

        Args:
            boxes: Bounding boxes in model input space (N, 4) [x1, y1, x2, y2]

        Returns:
            Bounding boxes in original image space
        """
        if len(boxes) == 0:
            return boxes

        restored_boxes = boxes.copy()

        # Remove padding offset
        restored_boxes[:, [0, 2]] -= self.pad_x  # x coordinates
        restored_boxes[:, [1, 3]] -= self.pad_y  # y coordinates

        # Scale back to original size
        restored_boxes = restored_boxes / self.scale_factor

        # Clip to original image bounds
        restored_boxes[:, [0, 2]] = np.clip(
            restored_boxes[:, [0, 2]], 0, self.original_width
        )
        restored_boxes[:, [1, 3]] = np.clip(
            restored_boxes[:, [1, 3]], 0, self.original_height
        )

        return restored_boxes

    def prepare_input(self, image):
        # Store original dimensions for coordinate restoration

        # Rescale image with padding instead of simple resize
        input_img = self.rescale_image_with_padding(
            image, (self.input_width, self.input_height)
        )

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        return np.expand_dims(input_img, 0).astype(np.float32)

    def inference(self, input_tensor):
        return self.session.run(self.output_names, {self.input_names[0]: input_tensor})

    def process_output(self, output):
        predictions = np.squeeze(output[0]).T

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], []

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = multiclass_nms(boxes, scores, class_ids, self.iou_threshold)

        # Restore coordinates to original image space
        final_boxes = self.restore_coordinates(boxes[indices])

        return final_boxes, scores[indices], class_ids[indices]

    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Convert boxes to xyxy format (no rescaling yet, done in restore_coordinates)
        return xywh2xyxy(boxes)

    def rescale_boxes(self, boxes):

        # Rescale boxes to original image dimensions
        input_shape = np.array(
            [self.input_width, self.input_height, self.input_width, self.input_height]
        )
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array(
            [self.img_width, self.img_height, self.img_width, self.img_height]
        )
        return boxes

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = (
            self.input_shape[2] if self.input_shape[2] != "height" else 960
        )
        self.input_width = (
            self.input_shape[3] if self.input_shape[3] != "width" else 960
        )

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]
