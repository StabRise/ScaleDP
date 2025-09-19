import gc
import logging
from types import MappingProxyType
from typing import Any

import numpy as np
from pyspark import keyword_only

from scaledp.enums import Device
from scaledp.models.detectors.BaseDetector import BaseDetector
from scaledp.params import HasBatchSize, HasDevice, HasModel, HasWhiteList
from scaledp.schemas.Box import Box
from scaledp.schemas.DetectorOutput import DetectorOutput


class LayoutDetector(BaseDetector, HasDevice, HasBatchSize, HasWhiteList, HasModel):
    _model = None

    defaultParams = MappingProxyType(
        {
            "inputCol": "image",
            "outputCol": "layout_boxes",
            "keepInputData": False,
            "scaleFactor": 1.0,
            "scoreThreshold": 0.5,
            "device": Device.CPU,
            "batchSize": 2,
            "partitionMap": False,
            "numPartitions": 0,
            "pageCol": "page",
            "pathCol": "path",
            "propagateError": False,
            "onlyRotated": False,
            "model": "PP-DocLayout_plus-L",
            "whiteList": [],
        },
    )

    @keyword_only
    def __init__(self, **kwargs: Any) -> None:
        super(LayoutDetector, self).__init__()
        self._setDefault(**self.defaultParams)
        self._set(**kwargs)
        self.get_model({k.name: v for k, v in self.extractParamMap().items()})

    @classmethod
    def get_model(cls, params):
        logging.info("Loading PaddleOCR LayoutDetection model...")
        if cls._model:
            return cls._model

        try:
            from paddleocr import LayoutDetection
        except ImportError as e:
            raise ImportError(
                "PaddleOCR is not installed. Please install it with: pip install paddleocr",
            ) from e

        # Get model name from params or use default
        model_name = params.get("model", "PP-DocLayout_plus-L")

        # Initialize LayoutDetection model
        device = "gpu" if int(params["device"]) == Device.CUDA.value else "cpu"
        cls._model = LayoutDetection(
            model_name=model_name,
            enable_hpi=False,
            device=device,
        )

        logging.info(
            f"PaddleOCR LayoutDetection model '{model_name}' loaded successfully",
        )
        return cls._model

    @classmethod
    def call_detector(cls, images, params):
        logging.info("Running LayoutDetector")

        detector = cls.get_model(params)
        layout_types = params.get("whiteList", [])

        logging.info("Process images for layout detection")
        results_final = []

        for image, image_path in images:
            boxes = []

            # Convert PIL to NumPy (RGB)
            image_np = np.array(image)

            try:
                # Run layout analysis using LayoutDetection
                result = detector.predict(input=image_np)

                if result and len(result) > 0:
                    # LayoutDetection returns a list of layout regions
                    result = result[0]
                    if isinstance(result, dict) and "boxes" in result:
                        for layout_item in result["boxes"]:
                            bbox = layout_item["coordinate"]  # Bounding box coordinates
                            layout_type = layout_item["label"]  # Layout type
                            confidence = layout_item.get("score", 1.0)

                            # Filter by layout type if specified
                            if layout_types and layout_type not in layout_types:
                                continue

                            # Filter by confidence threshold
                            if confidence < params["scoreThreshold"]:
                                continue

                            # Convert bbox to Box format
                            # LayoutDetection returns bbox as [x1, y1, x2, y2]
                            if len(bbox) == 4:
                                x = bbox[0]
                                y = bbox[1]
                                width = bbox[2] - bbox[0]
                                height = bbox[3] - bbox[1]

                                # Create Box with layout type as text
                                box = Box(
                                    text=layout_type,
                                    score=confidence,
                                    x=int(x),
                                    y=int(y),
                                    width=int(width),
                                    height=int(height),
                                )

                                # Add polygon points if needed for rotated boxes
                                if len(bbox) == 4:
                                    box.polygon = bbox

                                boxes.append(box)

            except Exception as e:
                logging.warning(f"Error in layout detection for {image_path}: {e!s}")
                if params.get("propagateError", False):
                    raise e

            if params.get("onlyRotated", False):
                boxes = [box for box in boxes if box.is_rotated()]

            results_final.append(
                DetectorOutput(path=image_path, type="layout", bboxes=boxes),
            )

        gc.collect()

        return results_final
