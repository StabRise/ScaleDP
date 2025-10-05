import gc
import logging
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar

import numpy as np
from huggingface_hub import hf_hub_download
from pyspark import keyword_only
from pyspark.ml.param import Param, Params, TypeConverters

from scaledp.enums import Device
from scaledp.models.detectors.BaseDetector import BaseDetector
from scaledp.models.detectors.yolo.yolo import YOLO
from scaledp.params import HasBatchSize, HasDevice
from scaledp.schemas.Box import Box
from scaledp.schemas.DetectorOutput import DetectorOutput


class YoloOnnxDetector(BaseDetector, HasDevice, HasBatchSize):
    _model: ClassVar = {}

    task = Param(
        Params._dummy(),
        "task",
        "Yolo task type.",
        typeConverter=TypeConverters.toString,
    )

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
            "task": "detect",
            "onlyRotated": False,
        },
    )

    @keyword_only
    def __init__(self, **kwargs: Any) -> None:
        super(YoloOnnxDetector, self).__init__()
        self._setDefault(**self.defaultParams)
        self._set(**kwargs)
        self.get_model({k.name: v for k, v in self.extractParamMap().items()})

    @classmethod
    def get_model(cls, params):

        model_path = params["model"]

        logging.info("Loading model...")
        if cls._model and model_path in cls._model:
            return cls._model.get(model_path)

        model_path_final = model_path
        if not Path(model_path).is_file():
            model_path_final = hf_hub_download(
                repo_id=model_path,
                filename="model.onnx",
            )

        logging.info("Model downloaded")

        detector = YOLO(model_path_final, params["scoreThreshold"])

        cls._model[model_path] = detector
        return cls._model[model_path]

    @classmethod
    def call_detector(cls, images, params):
        logging.info("Running YoloOnnxDetector")
        detector = cls.get_model(params)

        logging.info("Process images")
        print("process")
        results_final = []
        for image, image_path in images:
            boxes = []
            # Convert PIL to NumPy (RGB)
            image_np = np.array(image)
            raw_boxes, scores, class_ids = detector.detect_objects(image_np)
            print("detected")
            for box in raw_boxes:
                boxes.append(Box.from_bbox(box))
            results_final.append(
                DetectorOutput(path=image_path, type="yolo", bboxes=boxes),
            )

        gc.collect()

        return results_final
