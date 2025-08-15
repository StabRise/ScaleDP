import logging
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
from doctr.models import detection_predictor
from pyspark import keyword_only

from scaledp.enums import Device
from scaledp.models.detectors.BaseDetector import BaseDetector
from scaledp.params import HasBatchSize, HasDevice
from scaledp.schemas.Box import Box
from scaledp.schemas.DetectorOutput import DetectorOutput


class FastTextDetector(BaseDetector, HasDevice, HasBatchSize):
    _model = None
    defaultParams = MappingProxyType(
        {
            "inputCol": "image",
            "outputCol": "boxes",
            "keepInputData": False,
            "scaleFactor": 1.0,
            "scoreThreshold": 0.7,
            "batchSize": 2,
            "device": Device.CPU,
            "partitionMap": False,
            "numPartitions": 0,
            "pageCol": "page",
            "pathCol": "path",
            "propagateError": False,
            "onlyRotated": False,
        },
    )

    @keyword_only
    def __init__(self, **kwargs: Any) -> None:
        super(FastTextDetector, self).__init__()
        self._setDefault(**self.defaultParams)
        self._set(**kwargs)
        self.get_model({k.name: v for k, v in self.extractParamMap().items()})

    @classmethod
    def get_model(cls, params):
        if cls._model:
            return cls._model
        device = "cuda" if int(params["device"]) == Device.CUDA.value else "cpu"

        # Initialize the fast text detection model from doctr
        cls._model = detection_predictor(
            arch="fast_tiny",  # Use DBNet with ResNet50 backbone
            pretrained=True,  # Use pretrained weights
            assume_straight_pages=True,
        ).to(device)

        return cls._model

    @classmethod
    def call_detector(cls, images, params):
        model = cls.get_model(params)
        device = "cuda" if int(params["device"]) == Device.CUDA.value else "cpu"
        results = []

        for img, image_path in images:
            try:
                # Convert PIL image to numpy array if needed
                image = img
                if not isinstance(image, np.ndarray):
                    image = np.array(image)

                # Normalize image
                if image.max() > 1:
                    image = image / 255.0

                # Get predictions
                out = model([[image]])

                # Extract boxes from predictions
                # doctr returns relative coordinates (0-1)
                predictions = out[0]

                # Convert doctr boxes to Box objects
                # Scale relative coordinates to absolute coordinates
                height, width = image.shape[:2]
                box_objects = []
                for pred in predictions:
                    # Get coordinates
                    x_min, y_min = pred[0]  # Top-left
                    x_max, y_max = pred[2]  # Bottom-right

                    # Convert to absolute coordinates
                    abs_coords = [
                        int(x_min * width),  # x_min
                        int(y_min * height),  # y_min
                        int(x_max * width),  # x_max
                        int(y_max * height),  # y_max
                    ]

                    box_objects.append(Box.from_bbox(abs_coords))
                results.append(
                    DetectorOutput(
                        path=image_path,
                        type="fast",
                        bboxes=box_objects,
                        exception="",
                    ),
                )

            except Exception as e:
                logging.exception(e)
                results.append(
                    DetectorOutput(
                        path=image_path,
                        type="fast",
                        bboxes=[],
                        exception=f"FastTextDetector error: {e!s}",
                    ),
                )

        if device == "cuda":
            torch.cuda.empty_cache()

        return results
