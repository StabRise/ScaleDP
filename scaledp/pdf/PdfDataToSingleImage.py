import logging
from types import MappingProxyType
from typing import Any

import fitz
from pyspark import keyword_only

from scaledp.enums import ImageType
from scaledp.image.DataToImage import DataToImage
from scaledp.params import HasResolution, Param, Params, TypeConverters
from scaledp.schemas.Image import Image


class PdfDataToSingleImage(DataToImage, HasResolution):
    pageNumber = Param(
        Params._dummy(),
        "pageNumber",
        "Page number to convert to image",
        typeConverter=TypeConverters.toInt,
    )

    DEFAULT_PARAMS = MappingProxyType(
        {
            "inputCol": "content",
            "outputCol": "image",
            "pathCol": "path",
            "keepInputData": False,
            "imageType": ImageType.FILE,
            "resolution": 300,
            "pageNumber": 0,
        },
    )

    @keyword_only
    def __init__(self, **kwargs: Any) -> None:
        super(PdfDataToSingleImage, self).__init__()
        self._setDefault(**self.DEFAULT_PARAMS)
        self._set(**kwargs)

    def process(self, input, path, resolution=0):
        logging.info("Run Pdf to Image")
        try:
            doc = fitz.open("pdf", input)
            if len(doc) == 0:
                raise ValueError("Empty PDF document.")

            pix = doc[self.getPageNumber()].get_pixmap(
                matrix=fitz.Identity,
                dpi=self.getResolution(),
                colorspace=fitz.csRGB,
                clip=None,
                alpha=False,
                annots=True,
            )

            return Image.from_binary(
                pix.pil_tobytes("png"),
                path,
                self.getImageType(),
                resolution=self.getResolution(),
                width=pix.width,
                height=pix.height,
            )

        except Exception:
            return Image(
                path,
                exception="Error in extration of image from pdf document",
            )

    def getPageNumber(self):
        return self.getOrDefault(self.pageNumber)

    def setPageNumber(self, value):
        return self._set(pageNumber=value)
