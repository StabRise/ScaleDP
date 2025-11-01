import gc
import logging
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from huggingface_hub import hf_hub_download
from pyspark import keyword_only
from pyspark.ml.param import Param, Params, TypeConverters

from scaledp.enums import Device
from scaledp.models.detectors.BaseDetector import BaseDetector
from scaledp.params import HasBatchSize, HasDevice
from scaledp.schemas.Box import Box
from scaledp.schemas.DetectorOutput import DetectorOutput


class YoloOnnxTextDetector(BaseDetector, HasDevice, HasBatchSize):
    """YOLO ONNX text detector."""

    _model = None

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
        super(YoloOnnxTextDetector, self).__init__()
        self._setDefault(**self.defaultParams)
        self._set(**kwargs)
        self.get_model({k.name: v for k, v in self.extractParamMap().items()})

    @classmethod
    def get_model(cls, params):

        logging.info("Loading model...")
        if cls._model:
            return cls._model

        import onnxruntime as ort

        model = params["model"]
        if not Path(model).is_file():
            model = hf_hub_download(repo_id=model, filename="model.onnx")

        logging.info("Model downloaded")

        detector = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
        logging.info("Ort session created")

        cls._model = detector
        return cls._model

    @classmethod
    def obb_iou(cls, rect1, rect2):
        import cv2

        int_pts = cv2.rotatedRectangleIntersection(rect1, rect2)[1]
        if int_pts is None:
            return 0.0
        int_area = cv2.contourArea(int_pts)
        area1 = rect1[1][0] * rect1[1][1]
        area2 = rect2[1][0] * rect2[1][1]
        union = area1 + area2 - int_area
        return int_area / union if union > 0 else 0.0

    @classmethod
    def obb_nms(cls, boxes, iou_thresh=0.3):
        # boxes: [N, 6] -> cx, cy, w, h, conf, angle
        boxes = sorted(boxes, key=lambda x: x[4], reverse=True)  # sort by confidence
        keep = []

        while boxes:
            current = boxes.pop(0)
            keep.append(current)

            current_rect = (
                (current[0], current[1]),
                (current[2], current[3]),
                np.degrees(current[5]),
            )

            remaining = []
            for box in boxes:
                other_rect = (
                    (box[0], box[1]),
                    (box[2], box[3]),
                    np.degrees(box[5]),
                )
                iou = cls.obb_iou(current_rect, other_rect)
                if iou < iou_thresh:
                    remaining.append(box)
            boxes = remaining

        return np.array(keep)

    @classmethod
    def call_detector(cls, images, params):
        logging.info("Running YoloOnnxDetector")
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

            target_h = 960
            target_w = 960

            scale = min(target_w / image_rgb.shape[1], target_h / image_rgb.shape[0])
            resize_h = int(image_rgb.shape[0] * scale)
            resize_w = int(image_rgb.shape[1] * scale)
            img = cv2.resize(image_rgb, (resize_w, resize_h))

            if resize_h < target_h:
                pad_bottom = target_h - resize_h
                image_resized = cv2.copyMakeBorder(
                    img,
                    0,
                    pad_bottom,
                    0,
                    0,
                    cv2.BORDER_CONSTANT,
                    value=(255, 255, 255),
                )
            if resize_w < target_w:
                pad_right = target_w - resize_w
                image_resized = cv2.copyMakeBorder(
                    img,
                    0,
                    0,
                    0,
                    pad_right,
                    cv2.BORDER_CONSTANT,
                    value=(255, 255, 255),
                )
            ratio_h = ratio_w = scale

            # Preprocess
            input_tensor = image_resized.astype(np.float32) / 255.0  # Normalize
            input_tensor = np.transpose(input_tensor, (2, 0, 1))  # HWC -> CHW
            input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dim

            input_name = detector.get_inputs()[0].name

            logging.info("Run ort inference")

            # Inference
            outputs = detector.run(None, {input_name: input_tensor})

            logging.info("Complete ort inference")

            raw = outputs[0].squeeze(0)  # shape: (6, 8400)

            # Step 2: Transpose to (8400, 6)
            output = raw.T  # shape: (8400, 6)

            # Step 3: Confidence mask
            mask = output[:, 4] > params["scoreThreshold"]
            filtered = output[mask]

            nms_result = cls.obb_nms(filtered, iou_thresh=0.2)

            if params["task"] == "obb":

                for obb in nms_result:
                    cx, cy, bw, bh, conf, angle = obb

                    # Scale to original image if needed (skip if already absolute)
                    angle_scaled = float(
                        np.degrees(angle),
                    )  # Convert radians to degrees
                    box = ((float(cx), float(cy)), (float(bw), float(bh)), angle_scaled)
                    points = cv2.boxPoints(box)
                    points[:, 0] *= ratio_w  # x coordinates
                    points[:, 1] *= ratio_h  # y coordinates
                    points = np.intp(points)
                    boxes.append(Box.from_polygon(points))
            else:
                for box in nms_result:
                    boxes.append(Box.from_bbox(box.xyxy[0]))
            results_final.append(
                DetectorOutput(path=image_path, type="yolo", bboxes=boxes),
            )

        gc.collect()

        return results_final
