import logging
from types import MappingProxyType
from typing import Any

import cv2
import numpy as np
from pyspark import keyword_only
from pyspark.ml.param import Param, Params, TypeConverters

from scaledp.models.detectors.HasDetectLineOrientation import HasDetectLineOrientation
from scaledp.params import CODE_TO_LANGUAGE, LANGUAGE_TO_TESSERACT_CODE
from scaledp.schemas.Box import Box
from scaledp.schemas.Document import Document

from ...enums import OEM, PSM, TessLib
from .BaseRecognizer import BaseRecognizer


class TesseractRecognizer(BaseRecognizer, HasDetectLineOrientation):
    """
    Run Tesseract text recognition on images.
    """

    oem = Param(
        Params._dummy(),
        "oem",
        "OCR engine mode. Defaults to :attr:`OEM.DEFAULT`.",
        typeConverter=TypeConverters.toInt,
    )

    tessDataPath = Param(
        Params._dummy(),
        "tessDataPath",
        "Path to tesseract data folder.",
        typeConverter=TypeConverters.toString,
    )

    tessLib = Param(
        Params._dummy(),
        "tessLib",
        "The desired Tesseract library to use. Defaults to :attr:`TESSEROCR`",
        typeConverter=TypeConverters.toInt,
    )

    onlyRotated = Param(
        Params._dummy(),
        "onlyRotated",
        "Return only rotated boxes.",
        typeConverter=TypeConverters.toBoolean,
    )

    defaultParams = MappingProxyType(
        {
            "inputCols": ["image", "boxes"],
            "outputCol": "text",
            "keepInputData": False,
            "scaleFactor": 1.0,
            "scoreThreshold": 0.5,
            "oem": OEM.DEFAULT,
            "lang": ["eng"],
            "lineTolerance": 0,
            "keepFormatting": False,
            "tessDataPath": "/usr/share/tesseract-ocr/5/tessdata/",
            "tessLib": TessLib.PYTESSERACT,
            "partitionMap": False,
            "numPartitions": 0,
            "pageCol": "page",
            "pathCol": "path",
            "detectLineOrientation": True,
            "onlyRotated": True,
            "oriModel": "StabRise/line_orientation_detection_v0.1",
        },
    )

    @keyword_only
    def __init__(self, **kwargs: Any) -> None:
        super(TesseractRecognizer, self).__init__()
        self._setDefault(**self.defaultParams)
        self._set(**kwargs)

    @staticmethod
    def getLangTess(params):
        return "+".join(
            [
                LANGUAGE_TO_TESSERACT_CODE[CODE_TO_LANGUAGE[lang]]
                for lang in params["lang"]
            ],
        )

    @classmethod
    def _prepare_box_for_ocr(cls, image_np, box, params):

        scaled_box = box.scale(params["scaleFactor"], padding=5)

        center_tuple = (
            scaled_box.x + scaled_box.width / 2,
            scaled_box.y + scaled_box.height / 2,
        )
        size_tuple = (scaled_box.width, scaled_box.height)

        rect = (center_tuple, size_tuple, scaled_box.angle)
        src_pts = cv2.boxPoints(rect).astype("float32")
        dst_pts = np.array(
            [
                [0, int(scaled_box.height) - 1],
                [0, 0],
                [int(scaled_box.width) - 1, 0],
                [int(scaled_box.width) - 1, int(scaled_box.height) - 1],
            ],
            dtype="float32",
        )

        try:
            m_transform = cv2.getPerspectiveTransform(src_pts, dst_pts)
            return cv2.warpPerspective(
                image_np,
                m_transform,
                (int(scaled_box.width), int(scaled_box.height)),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )
        except cv2.error as e:
            logging.error(f"Error processing box {box}: {e}")
            return None

    @classmethod
    def _convert_to_pil(cls, cropped_np):
        from PIL import Image

        if cropped_np is None or cropped_np.size == 0:
            return None
        if cropped_np.ndim == 3:
            if cropped_np.shape[2] == 4:
                cropped_np = cv2.cvtColor(cropped_np, cv2.COLOR_RGBA2RGB)
            elif cropped_np.shape[2] not in (3,):
                cropped_np = cv2.cvtColor(cropped_np, cv2.COLOR_BGR2GRAY)
        elif cropped_np.ndim != 2:
            return None
        return Image.fromarray(cropped_np)

    @classmethod
    def _process_image_with_tesseract(cls, image, image_path, detected_boxes, params):
        from tesserocr import PSM, PyTessBaseAPI

        boxes, texts = [], []
        image_np = np.array(image)
        lang = cls.getLangTess(params)

        with PyTessBaseAPI(
            path=params["tessDataPath"],
            psm=PSM.SINGLE_WORD,
            oem=params["oem"],
            lang=lang,
        ) as api:
            api.SetVariable("debug_file", "ocr.log")
            for box_raw in detected_boxes.bboxes:
                # Ensure box is Box instance
                if isinstance(box_raw, dict):
                    box = Box(**box_raw)
                elif not isinstance(box_raw, Box):
                    box = Box(**box_raw.asDict())
                else:
                    box = box_raw
                cropped_np = cls._prepare_box_for_ocr(image_np, box, params)
                # Auto-orient the image before OCR

                pil_image = cls._convert_to_pil(cropped_np)
                if params["detectLineOrientation"]:
                    pil_image, orientation = cls.auto_orient_image(pil_image, params)
                if pil_image is None:
                    continue

                if (
                    params["onlyRotated"]
                    and not box.is_rotated()
                    and orientation != "180_degree"
                ):
                    continue

                api.SetImage(pil_image)
                api.Recognize(0)
                b = box
                if isinstance(box, dict):
                    b = Box(**b)
                b.text = api.GetUTF8Text()
                b.conf = api.MeanTextConf()
                if b.score > params["scoreThreshold"]:
                    boxes.append(b)
                    texts.append(b.text)

        if params["keepFormatting"]:
            text = TesseractRecognizer.box_to_formatted_text(
                boxes,
                params["lineTolerance"],
            )
        else:
            text = " ".join(texts)

        return Document(
            path=image_path,
            text=text,
            bboxes=boxes,
            type="text",
            exception="",
        )

    @classmethod
    def call_pytesseract(cls, images, detected_boxes, params):  # pragma: no cover
        results = []
        for (image, image_path), boxes in zip(images, detected_boxes):
            doc = cls._process_image_with_tesseract(image, image_path, boxes, params)
            results.append(doc)
        return results

    @classmethod
    def call_tesserocr(cls, images, detected_boxes, params):  # pragma: no cover
        from tesserocr import PyTessBaseAPI

        results = []
        lang = cls.getLangTess(params)
        with PyTessBaseAPI() as api:
            api.Init(params["tessDataPath"], lang, oem=params["oem"])
            api.SetPageSegMode(PSM.SINGLE_WORD)
            api.SetVariable("debug_file", "ocr.log")

            for (image, image_path), detected_box in zip(images, detected_boxes):
                api.SetImage(image)

                boxes = []
                texts = []

                for b in detected_box.bboxes:
                    box = b
                    if isinstance(box, dict):
                        box = Box(**box)
                    if not isinstance(box, Box):
                        box = Box(**box.asDict())
                    scaled_box = box.scale(params["scaleFactor"], padding=0)
                    api.SetRectangle(
                        scaled_box.x,
                        scaled_box.y,
                        scaled_box.width,
                        scaled_box.height,
                    )
                    box.text = api.GetUTF8Text().replace("\n", "")
                    box.score = api.MeanTextConf() / 100
                    if box.score > params["scoreThreshold"]:
                        boxes.append(box)
                        texts.append(box.text)
                if params["keepFormatting"]:
                    text = TesseractRecognizer.box_to_formatted_text(
                        boxes,
                        params["lineTolerance"],
                    )
                else:
                    text = " ".join(texts)

                results.append(
                    Document(
                        path=image_path,
                        text=text,
                        bboxes=boxes,
                        type="text",
                        exception="",
                    ),
                )
        return results

    @classmethod
    def call_recognizer(cls, images, boxes, params):
        if params["tessLib"] == TessLib.TESSEROCR.value:
            return cls.call_tesserocr(images, boxes, params)
        if params["tessLib"] == TessLib.PYTESSERACT.value:
            return cls.call_pytesseract(images, boxes, params)
        raise ValueError(f"Unknown Tesseract library: {params['tessLib']}")

    def setOem(self, value):
        """
        Sets the value of :py:attr:`oem`.
        """
        return self._set(oem=value)

    def getOem(self):
        """
        Sets the value of :py:attr:`oem`.
        """
        return self.getOrDefault(self.oem)

    def setTessDataPath(self, value):
        """
        Sets the value of :py:attr:`tessDataPath`.
        """
        return self._set(tessDataPath=value)

    def getTessDataPath(self):
        """
        Sets the value of :py:attr:`tessDataPath`.
        """
        return self.getOrDefault(self.tessDataPath)

    def setTessLib(self, value):
        """
        Sets the value of :py:attr:`tessLib`.
        """
        return self._set(tessLib=value)

    def getTessLib(self):
        """
        Gets the value of :py:attr:`tessLib`.
        """
        return self.getOrDefault(self.tessLib)
