import gc
import io
from types import MappingProxyType
from typing import Any

from pyspark import keyword_only

from scaledp.enums import Device
from scaledp.models.recognizers.BaseOcr import BaseOcr
from scaledp.params import HasBatchSize, HasDevice
from scaledp.schemas.Box import Box
from scaledp.schemas.Document import Document


class DocTROcr(BaseOcr, HasDevice, HasBatchSize):

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
        super(DocTROcr, self).__init__()
        self._setDefault(**self.defaultParams)
        self._set(**kwargs)

    @classmethod
    def call_ocr(cls, images, params):
        import torch
        from doctr.io import DocumentFile
        from doctr.models import ocr_predictor

        model = ocr_predictor(pretrained=True)
        results = []
        for image, image_path in images:
            buff = io.BytesIO()
            image.save(buff, "png")
            doc = DocumentFile.from_images(buff.getvalue())
            # Analyze
            result = model(doc)

            boxes = []

            for page in result.pages:
                h, w = page.dimensions
                for block in page.blocks:
                    for line in block.lines:
                        for word in line.words:

                            xmin, ymin, xmax, ymax = [
                                tupl
                                for tuploftupls in word.geometry
                                for tupl in tuploftupls
                            ]
                            xmin = int(xmin * w)
                            ymin = int(ymin * h)
                            xmax = int(xmax * w)
                            ymax = int(ymax * h)

                            boxes.append(
                                Box(
                                    word.value,
                                    word.confidence,
                                    xmin,
                                    ymin,
                                    abs(xmax - xmin),
                                    abs(ymax - ymin),
                                ),
                            )

            if params["keepFormatting"]:
                text = DocTROcr.box_to_formatted_text(boxes, params["lineTolerance"])
            else:
                text = result.render()

            results.append(
                Document(path=image_path, text=text, type="text", bboxes=boxes),
            )

        gc.collect()

        if int(params["device"]) == Device.CUDA.value:
            torch.cuda.empty_cache()

        return results
