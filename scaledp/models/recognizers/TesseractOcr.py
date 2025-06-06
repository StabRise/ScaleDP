from types import MappingProxyType
from typing import Any

from pyspark import keyword_only

from scaledp.models.recognizers.BaseOcr import BaseOcr
from scaledp.params import (
    CODE_TO_LANGUAGE,
    LANGUAGE_TO_TESSERACT_CODE,
    Param,
    Params,
    TypeConverters,
)
from scaledp.schemas.Box import Box
from scaledp.schemas.Document import Document

from ...enums import OEM, PSM, TessLib


class TesseractOcr(BaseOcr):
    """
    Run Tesseract OCR text recognition on images.
    """

    psm = Param(
        Params._dummy(),
        "psm",
        "The desired PageSegMode. Defaults to :attr:`PSM.AUTO",
        typeConverter=TypeConverters.toInt,
    )

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

    defaultParams = MappingProxyType(
        {
            "inputCol": "image",
            "outputCol": "text",
            "bypassCol": "",
            "keepInputData": False,
            "scaleFactor": 1.0,
            "scoreThreshold": 0.5,
            "psm": PSM.AUTO,
            "oem": OEM.DEFAULT,
            "lang": ["eng"],
            "lineTolerance": 0,
            "keepFormatting": False,
            "tessDataPath": "/usr/share/tesseract-ocr/5/tessdata/",
            "tessLib": TessLib.PYTESSERACT,
            "partitionMap": False,
            "propagateError": False,
        },
    )

    @keyword_only
    def __init__(self, **kwargs: Any) -> None:
        super(TesseractOcr, self).__init__()
        self._setDefault(**self.defaultParams)
        self._set(**kwargs)

    @classmethod
    def call_pytesseract(cls, images, params):
        import pytesseract

        results = []

        config = (
            f"--psm {params['psm']} --oem {params['oem']} -l {cls.getLangTess(params)}"
        )
        for image, image_path in images:
            res = pytesseract.image_to_data(
                image,
                output_type=pytesseract.Output.DATAFRAME,
                config=config,
            )
            res["conf"] = res["conf"] / 100

            if not params["keepFormatting"]:
                res.loc[res["level"] == 4, "conf"] = 1.0
                res["text"] = res["text"].fillna("\n")

            res = res[res["conf"] > params["scoreThreshold"]][
                ["text", "conf", "left", "top", "width", "height"]
            ].rename(columns={"conf": "score", "left": "x", "top": "y"})
            res = res[res["text"] != "\n"]
            boxes = res.apply(
                lambda x: Box(*x).to_string().scale(1 / params["scaleFactor"]),
                axis=1,
            ).values.tolist()
            if params["keepFormatting"]:
                text = TesseractOcr.box_to_formatted_text(
                    boxes,
                    params["lineTolerance"],
                )
            else:
                text = " ".join([str(w) for w in res["text"].values.tolist()])

            results.append(
                Document(path=image_path, text=text, type="text", bboxes=boxes),
            )
        return results

    @staticmethod
    def getLangTess(params):
        return "+".join(
            [
                LANGUAGE_TO_TESSERACT_CODE[CODE_TO_LANGUAGE[lang]]
                for lang in params["lang"]
            ],
        )

    @classmethod
    def call_tesserocr(cls, images, params):  # pragma: no cover
        from tesserocr import RIL, PyTessBaseAPI, iterate_level

        results = []

        with PyTessBaseAPI(
            path=params["tessDataPath"],
            psm=params["psm"],
            oem=params["oem"],
            lang=cls.getLangTess(params),
        ) as api:
            api.SetVariable("debug_file", "ocr.log")
            api.SetVariable("save_blob_choices", "T")

            for image, image_path in images:
                api.SetImage(image)

                api.Recognize()
                iterator = api.GetIterator()
                boxes = []
                texts = []

                level = RIL.WORD
                for r in iterate_level(iterator, level):
                    conf = r.Confidence(level) / 100
                    text = r.GetUTF8Text(level)
                    box = r.BoundingBox(level)
                    if conf > params["scoreThreshold"]:
                        boxes.append(
                            Box(
                                text,
                                conf,
                                box[0],
                                box[1],
                                abs(box[2] - box[0]),
                                abs(box[3] - box[1]),
                            ).scale(1 / params["scaleFactor"]),
                        )
                        texts.append(text)
                if params["keepFormatting"]:
                    text = TesseractOcr.box_to_formatted_text(
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
    def call_ocr(cls, images, params):
        if params["tessLib"] == TessLib.TESSEROCR.value:
            return cls.call_tesserocr(images, params)
        if params["tessLib"] == TessLib.PYTESSERACT.value:
            return cls.call_pytesseract(images, params)
        raise ValueError(f"Unknown Tesseract library: {params['tessLib']}")

    def setPsm(self, value):
        """
        Sets the value of :py:attr:`psm`.
        """
        return self._set(psm=value)

    def getPsm(self):
        """
        Sets the value of :py:attr:`psm`.
        """
        return self.getOrDefault(self.psm)

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
