import io
import logging
import traceback
from types import MappingProxyType
from typing import Any

from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql.functions import udf

from scaledp.params import HasInputCol, HasOutputCol, HasResolution
from scaledp.schemas.PdfDocument import PdfDocument


class SingleImageToPdf(
    Transformer,
    DefaultParamsReadable,
    DefaultParamsWritable,
    HasOutputCol,
    HasInputCol,
    HasResolution,
):
    """Transform Image to PDF."""

    POINTS_PER_INCH = 72

    DEFAULT_PARAMS = MappingProxyType(
        {
            "inputCol": "image",
            "outputCol": "pdf",
            "resolution": 0,
        },
    )

    @keyword_only
    def __init__(self, **kwargs: Any) -> None:
        super(SingleImageToPdf, self).__init__()
        self._setDefault(**self.DEFAULT_PARAMS)
        self._set(**kwargs)

    def transform_udf(self, image):
        try:
            if image.exception != "":
                return PdfDocument(
                    path=image.path,
                    data=bytes(),
                    exception=image.exception,
                )
            image_resolution = image.resolution
            if image_resolution == 0 or image_resolution is None:
                image_resolution = 300
            if self.getResolution() > 0:
                image_resolution = self.getResolution()

            width = image.width / image_resolution * self.POINTS_PER_INCH
            height = image.height / image_resolution * self.POINTS_PER_INCH

            import img2pdf

            a4inpt = (img2pdf.mm_to_pt(210), img2pdf.mm_to_pt(297))
            layout_fun = img2pdf.get_layout_fun(a4inpt)
            pdf_bytes = img2pdf.convert(io.BytesIO(image.data), layout_fun=layout_fun)
        except Exception:
            exception = traceback.format_exc()
            exception = f"SingleImageToPdf: {exception}, {image.exception}"
            logging.warning(exception)
            return PdfDocument(path=image.path, data=bytes(), exception=exception)

        return PdfDocument(
            path=image.path,
            data=pdf_bytes,
            width=width,
            height=height,
            exception="",
        )

    def _transform(self, dataset):

        output_col = self.getOutputCol()
        input_col = self.getInputCol()

        if input_col not in dataset.columns:
            raise ValueError(
                f"Input column '{input_col}' is not present in the DataFrame.",
            )

        input_col = dataset[self.getInputCol()]

        return dataset.withColumn(
            output_col,
            udf(self.transform_udf, PdfDocument.get_schema())(input_col),
        )
