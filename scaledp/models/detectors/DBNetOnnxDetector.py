import gc
import logging
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from huggingface_hub import hf_hub_download
from pyspark import keyword_only

from scaledp.enums import Device
from scaledp.models.detectors.BaseDetector import BaseDetector
from scaledp.models.detectors.paddle_onnx.predict_det import DBNetTextDetector
from scaledp.params import HasBatchSize, HasDevice
from scaledp.schemas.Box import Box
from scaledp.schemas.DetectorOutput import DetectorOutput


class DBNetOnnxDetector(BaseDetector, HasDevice, HasBatchSize):
    _model = None

    defaultParams = MappingProxyType(
        {
            "inputCol": "image",
            "outputCol": "boxes",
            "keepInputData": False,
            "scaleFactor": 1.0,
            "scoreThreshold": 0.2,
            "device": Device.CPU,
            "batchSize": 2,
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
        super(DBNetOnnxDetector, self).__init__()
        self._setDefault(**self.defaultParams)
        self._set(**kwargs)
        self.get_model({k.name: v for k, v in self.extractParamMap().items()})

    @classmethod
    def get_model(cls, params):

        logging.info("Loading model...")
        if cls._model:
            return cls._model

        model = params["model"]
        if not Path(model).is_file():
            model = hf_hub_download(repo_id=model, filename="model.onnx")

        logging.info("Model downloaded")

        detector = DBNetTextDetector(model, use_gpu=params["model"] == Device.CUDA)

        cls._model = detector
        return cls._model

    @classmethod
    def call_detector(cls, images, params):
        logging.info("Running DBNetOnnxDetector")
        import cv2

        detector = cls.get_model(params)

        logging.info("Process images")
        results_final = []
        for image, image_path in images:
            boxes = []

            # Convert PIL to NumPy (RGB)
            image_np = np.array(image)

            # Convert RGB to BGR for OpenCV
            image_rgb = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            result = detector(image_rgb)

            for points in result:
                boxes.append(Box.from_polygon(points, padding=0))

            if params["onlyRotated"]:
                boxes = [box for box in boxes if box.is_rotated()]

            # Merge overlapping boxes before returning, only if on the same line and similar angle
            boxes = Box.merge_overlapping_boxes(
                boxes,
                iou_threshold=0.02,  # as before
                angle_thresh=10.0,  # only merge if angle difference < 10 degrees
                line_thresh=0.3,  # only merge if centers are close (half
                # height)
            )

            results_final.append(
                DetectorOutput(path=image_path, type="DBNetOnnx", bboxes=boxes),
            )

        gc.collect()

        return results_final
