import gc
from types import MappingProxyType
from typing import Any

import numpy as np
from pyspark import keyword_only

from scaledp.enums import Device
from scaledp.models.recognizers.BaseOcr import BaseOcr
from scaledp.params import HasBatchSize, HasDevice
from scaledp.schemas.Box import Box
from scaledp.schemas.Document import Document


class EasyOcr(BaseOcr, HasDevice, HasBatchSize):

    defaultParams = MappingProxyType(
        {
            "inputCol": "image",
            "outputCol": "text",
            "bypassCol": "",
            "keepInputData": False,
            "scaleFactor": 1.0,
            "scoreThreshold": 0.5,
            "lang": ["eng"],
            "lineTolerance": 0,
            "keepFormatting": False,
            "partitionMap": False,
            "numPartitions": 0,
            "pageCol": "page",
            "pathCol": "path",
            "device": Device.CPU,
            "batchSize": 2,
            "propagateError": False,
        },
    )

    @keyword_only
    def __init__(self, **kwargs: Any) -> None:
        super(EasyOcr, self).__init__()
        self._setDefault(**self.defaultParams)
        self._set(**kwargs)

    @staticmethod
    def points_to_box(points, text, score):
        """Convert a set of four corner points to (x, y, width, height)."""
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]

        x = min(x_coords)
        y = min(y_coords)
        width = max(x_coords) - x
        height = max(y_coords) - y

        return Box(text=text, score=score, x=x, y=y, width=width, height=height)

    @classmethod
    def call_ocr(cls, images, params):
        import easyocr
        import torch

        device = int(params["device"]) != Device.CPU.value

        langs = params["lang"]
        scale_factor = params["scaleFactor"]
        reader = easyocr.Reader(langs, device)
        results = []
        for img, image_path in images:
            image = np.array(img.convert("RGB"))[:, :, ::-1].copy()
            result = reader.readtext(image)
            boxes = [
                EasyOcr.points_to_box(box, text, float(score))
                .to_string()
                .scale(1 / scale_factor)
                for box, text, score in result
            ]

            if params["keepFormatting"]:
                text = EasyOcr.box_to_formatted_text(boxes, params["lineTolerance"])
            else:
                text = "\n".join([str(w.text) for w in boxes])

            results.append(
                Document(path=image_path, text=text, type="text", bboxes=boxes),
            )

        gc.collect()
        if int(params["device"]) == Device.CUDA.value:
            torch.cuda.empty_cache()

        return results
