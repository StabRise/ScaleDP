import json
from pyspark import keyword_only

from ...enums import Device
from sparkpdf.params import *
from sparkpdf.schemas.Image import Image
from sparkpdf.schemas.Box import Box
from sparkpdf.schemas.Document import Document
from sparkpdf.models.recognizers.BaseOcr import BaseOcr
import pandas as pd
import gc
import torch

class SuryaOcr(BaseOcr, HasDevice, HasBatchSize):

    defaultParams = {
        "inputCol": "image",
        "outputCol": "text",
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
    }

    @keyword_only
    def __init__(self, **kwargs):
        super(SuryaOcr, self).__init__()
        self._setDefault(**self.defaultParams)
        self._set(**kwargs)

    @classmethod
    def call_ocr(cls, images, params):
        from surya.ocr import run_ocr
        from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
        from surya.model.recognition.model import load_model as load_rec_model
        from surya.model.recognition.processor import load_processor as load_rec_processor
        from surya.settings import settings

        if int(params['device']) == Device.CPU.value:
            device = "cpu"
        else:
            device = "cuda"

        langs = params["lang"]

        settings.DETECTOR_BATCH_SIZE = params["batchSize"]
        settings.RECOGNITION_BATCH_SIZE = params["batchSize"]

        det_processor, det_model = load_det_processor(), load_det_model(device=device)
        rec_model, rec_processor = load_rec_model(device=device), load_rec_processor()

        results = []
        predictions = run_ocr([image[0] for image in images], [langs] * len(images), det_model, det_processor, rec_model,
                              rec_processor)

        for prediction, (image, image_path) in zip(predictions, images):
            boxes = [Box(text=x.text, score=x.confidence, x=x.bbox[0],
                         y=x.bbox[1], width=x.bbox[2] - x.bbox[0], height=x.bbox[3] - x.bbox[1])
                     .toString().scale(1 / params["scaleFactor"]) for x in prediction.text_lines]

            if params["keepFormatting"]:
                text = SuryaOcr.box_to_formatted_text(boxes, params["lineTolerance"])
            else:
                text = "\n".join([str(w.text) for w in boxes])

            results.append(Document(path=image_path,
                                    text=text,
                                    type="text",
                                    bboxes=boxes))

        gc.collect()
        if int(params['device']) == Device.CUDA.value:
            torch.cuda.empty_cache()

        return results
