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
from scaledp.params import HasBatchSize, HasDevice, HasLabels
from scaledp.schemas.Box import Box
from scaledp.schemas.DetectorOutput import DetectorOutput


class YoloOnnxDetector(BaseDetector, HasDevice, HasBatchSize, HasLabels):
    """YOLO ONNX object detector."""

    _model: ClassVar = {}

    task = Param(
        Params._dummy(),
        "task",
        "Yolo task type.",
        typeConverter=TypeConverters.toString,
    )

    # Add padding param: integer percent to expand detected boxes
    padding = Param(
        Params._dummy(),
        "padding",
        "Padding percent to expand detected boxes (integer).",
        typeConverter=TypeConverters.toInt,
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
            "pageCol": "",
            "pathCol": "",
            "propagateError": False,
            "task": "detect",
            "onlyRotated": False,
            "padding": 0,  # default padding percent
            "labels": [],  # default empty labels
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

        detector = YOLO(
            model_path_final,
            conf_thres=params["scoreThreshold"],
            device=params["device"],
        )

        cls._model[model_path] = detector
        return cls._model[model_path]

    @classmethod
    def call_detector(cls, images, params):
        logging.info("Running YoloOnnxDetector")
        detector = cls.get_model(params)

        logging.info("Process images")
        results_final = []
        for image, image_path in images:
            boxes = []
            # Convert PIL to NumPy (RGB)
            image_np = np.array(image.convert("RGB"))
            raw_boxes, scores, class_ids = detector.detect_objects(image_np)
            # Expand boxes by padding percent if provided
            pad_percent = int(params.get("padding", 0)) if params is not None else 0
            h_img, w_img = image_np.shape[:2]
            labels = params.get("labels", [])
            for i, box in enumerate(raw_boxes):
                # Assume box format is [x1, y1, x2, y2]
                if pad_percent and len(box) >= 4:
                    x1, y1, x2, y2 = (
                        float(box[0]),
                        float(box[1]),
                        float(box[2]),
                        float(box[3]),
                    )
                    w = x2 - x1
                    h = y2 - y1
                    dx = (pad_percent / 100.0) * w
                    dy = (pad_percent / 100.0) * h
                    x1_new = max(0.0, x1 - dx)
                    y1_new = max(0.0, y1 - dy)
                    x2_new = min(float(w_img - 1), x2 + dx)
                    y2_new = min(float(h_img - 1), y2 + dy)
                    expanded_box = [x1_new, y1_new, x2_new, y2_new]
                else:
                    expanded_box = box
                # Map class_id to label and get score
                label = (
                    labels[class_ids[i]]
                    if labels and class_ids[i] < len(labels)
                    else str(class_ids[i])
                )
                score = scores[i] if scores is not None and i < len(scores) else 0.0
                boxes.append(Box.from_bbox(expanded_box, label=label, score=score))
            results_final.append(
                DetectorOutput(path=image_path, type="yolo", bboxes=boxes),
            )

        gc.collect()

        return results_final
