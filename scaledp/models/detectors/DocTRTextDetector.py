import gc
import io
from types import MappingProxyType
from typing import Any

from pyspark import keyword_only

from scaledp.models.detectors.BaseDetector import BaseDetector
from scaledp.params import HasBatchSize, HasDevice
from scaledp.schemas.Box import Box
from scaledp.schemas.DetectorOutput import DetectorOutput

from ...enums import Device


class DocTRTextDetector(BaseDetector, HasDevice, HasBatchSize):

    defaultParams = MappingProxyType(
        {
            "inputCol": "image",
            "outputCol": "boxes",
            "keepInputData": False,
            "scaleFactor": 1.0,
            "scoreThreshold": 0.1,
            "device": Device.CPU,
            "batchSize": 2,
            "model": "db_resnet50",
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
        super(DocTRTextDetector, self).__init__()
        self._setDefault(**self.defaultParams)
        self._set(**kwargs)

    @classmethod
    def call_detector(cls, images, params):
        import torch
        from doctr.file_utils import CLASS_NAME
        from doctr.io import DocumentFile
        from doctr.models import ocr_predictor

        predictor = ocr_predictor(pretrained=True, det_arch=params["model"])

        predictor.det_predictor.model.postprocessor.box_thresh = params[
            "scoreThreshold"
        ]

        detector = predictor.det_predictor

        device = "cpu" if int(params["device"]) == Device.CPU.value else "cuda"

        def resolve_geometry(
            geom: Any,
        ) -> (
            tuple[float, float, float, float]
            | tuple[float, float, float, float, float, float, float, float]
        ):
            if len(geom) == 4:
                return (*geom[0], *geom[1], *geom[2], *geom[3])
            return (*geom[0], *geom[1])

        docs = []
        for image, _image_path in images:
            buff = io.BytesIO()
            image.save(buff, "png")
            docs.extend(DocumentFile.from_images(buff.getvalue()))

        results = detector.to(device)(docs)

        results_final = []
        for result, (image, image_path) in zip(results, images):
            boxes = []
            w, h = image.size
            for geom in result[CLASS_NAME]:
                g = (
                    geom[:-1].tolist()
                    if geom.shape == (5,)
                    else resolve_geometry(geom[:4].tolist())
                )
                boxes.append(
                    Box.from_bbox(
                        [g[0] * w, g[1] * h, g[2] * w, g[3] * h],
                        score=geom[-1],
                    ),
                )
            results_final.append(
                DetectorOutput(path=image_path, type="doctr", bboxes=boxes),
            )

        gc.collect()
        if int(params["device"]) == Device.CUDA.value:
            torch.cuda.empty_cache()

        return results_final
