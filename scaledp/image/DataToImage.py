import logging
import traceback
from types import MappingProxyType
from typing import Any

from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql.functions import lit, udf

from scaledp.enums import ImageType
from scaledp.params import (
    HasColumnValidator,
    HasDefaultEnum,
    HasImageType,
    HasInputCol,
    HasKeepInputData,
    HasOutputCol,
    HasPathCol,
    HasPropagateExc,
)
from scaledp.schemas.Image import Image


class ImageError(Exception):
    pass


class DataToImage(
    Transformer,
    HasInputCol,
    HasOutputCol,
    HasKeepInputData,
    HasImageType,
    HasDefaultEnum,
    HasPathCol,
    DefaultParamsReadable,
    DefaultParamsWritable,
    HasColumnValidator,
    HasPropagateExc,
):
    """Transform Binary Content to Image."""

    defaultParams = MappingProxyType(
        {
            "inputCol": "content",
            "outputCol": "image",
            "pathCol": "path",
            "keepInputData": False,
            "imageType": ImageType.FILE,
            "propagateError": False,
        },
    )

    @keyword_only
    def __init__(self, **kwargs: Any) -> None:
        super(DataToImage, self).__init__()
        self._setDefault(**self.defaultParams)
        self._set(**kwargs)

    def process(self, input, path, resolution):
        return Image.from_binary(
            input,
            path,
            self.getImageType(),
            resolution=resolution,
        )

    def transform_udf(self, input, path, resolution):
        try:
            return self.process(input, path, resolution)
        except Exception as ex:
            exception = traceback.format_exc()
            exception = f"DataToImage: {exception}"
            logging.warning(exception)
            if self.getPropagateError():
                raise ImageError(exception) from ex
            return Image(path, self.getImageType(), data=bytes(), exception=exception)

    def _transform(self, dataset):
        out_col = self.getOutputCol()
        input_col = self._validate(self.getInputCol(), dataset)
        path_col = self._validate(self.getPathCol(), dataset)
        resolution = (
            dataset["resolution"] if "resolution" in dataset.columns else lit(0)
        )
        result = dataset.withColumn(
            out_col,
            udf(self.transform_udf, Image.get_schema())(
                input_col,
                path_col,
                resolution,
            ),
        )
        if not self.getKeepInputData():
            result = result.drop(self.getInputCol())
        return result
