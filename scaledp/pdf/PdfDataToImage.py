import logging
import traceback
from types import MappingProxyType
from typing import Any

import fitz
from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.pandas import DataFrame
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, Row

from scaledp.enums import ImageType
from scaledp.params import (
    HasColumnValidator,
    HasDefaultEnum,
    HasImageType,
    HasInputCol,
    HasKeepInputData,
    HasOutputCol,
    HasPageCol,
    HasPathCol,
    HasResolution,
    Param,
    Params,
    TypeConverters,
)
from scaledp.pipeline.PandasPipeline import posexplode
from scaledp.schemas.Image import Image


class PdfDataToImage(
    Transformer,
    HasInputCol,
    HasOutputCol,
    HasKeepInputData,
    HasImageType,
    HasPathCol,
    HasResolution,
    HasPageCol,
    DefaultParamsReadable,
    DefaultParamsWritable,
    HasColumnValidator,
    HasDefaultEnum,
):
    """Extract image from PDF file."""

    pageLimit = Param(
        Params._dummy(),
        "pageLimit",
        "Limit number of pages to convert to image",
        typeConverter=TypeConverters.toInt,
    )

    DEFAULT_PARAMS = MappingProxyType(
        {
            "inputCol": "content",
            "outputCol": "image",
            "pathCol": "path",
            "pageCol": "page",
            "keepInputData": False,
            "imageType": ImageType.FILE,
            "resolution": 300,
            "pageLimit": 0,
        },
    )

    @keyword_only
    def __init__(self, **kwargs: Any) -> None:
        super(PdfDataToImage, self).__init__()
        self._setDefault(**self.DEFAULT_PARAMS)
        self._set(**kwargs)

    def transform_udf(self, input: Row, path: Row) -> list[Image]:
        logging.info("Run Pdf Data to Image")
        try:
            doc = fitz.open("pdf", input)
            if len(doc) == 0:
                raise ValueError("Empty PDF document.")
            if self.getPageLimit():
                doc = doc[: self.getPageLimit()]
            for page in doc:
                pix = page.get_pixmap(
                    matrix=fitz.Identity,
                    dpi=self.getResolution(),
                    colorspace=fitz.csRGB,
                    clip=None,
                    alpha=False,
                    annots=True,
                )
                yield Image.from_binary(
                    pix.pil_tobytes("png"),
                    path,
                    self.getImageType(),
                    resolution=self.getResolution(),
                    width=pix.width,
                    height=pix.height,
                )

        except Exception:
            exception = traceback.format_exc()
            exception = (
                f"{self.uid}: Error during extract image from "
                f"the PDF document: {exception}"
            )
            logging.warning(exception)
            return [Image(path=path, exception=exception)]

    def _transform(self, dataset: DataFrame) -> DataFrame:
        out_col = self.getOutputCol()
        input_col = self._validate(self.getInputCol(), dataset)
        path_col = dataset[self.getPathCol()]

        df_1 = dataset.withColumn(
            "temp_data",
            udf(self.transform_udf, ArrayType(Image.get_schema()))(
                input_col,
                path_col,
            ),
        )

        result = posexplode(df_1, "temp_data", self.getPageCol(), out_col)

        if not self.getKeepInputData():
            result = result.drop(self.getInputCol())
        return result

    def getPageLimit(self) -> int:
        """Get page limit."""
        return self.getOrDefault(self.pageLimit)

    def setPageLimit(self, value) -> "PdfDataToImage":
        """Set page limit."""
        return self._set(pageNumber=value)
