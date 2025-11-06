import logging
import traceback
from types import MappingProxyType
from typing import Any

from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql.functions import explode, udf
from pyspark.sql.types import ArrayType

from scaledp.params import (
    AutoParamsMeta,
    HasColor,
    HasColumnValidator,
    HasDefaultEnum,
    HasImageType,
    HasInputCols,
    HasKeepInputData,
    HasNumPartitions,
    HasOutputCol,
    HasPageCol,
    HasPropagateExc,
    Param,
    Params,
    TypeConverters,
)
from scaledp.schemas.Box import Box
from scaledp.schemas.Image import Image

from ..enums import ImageType


class ImageCropError(Exception):
    pass


class ImageCropBoxes(
    Transformer,
    HasInputCols,
    HasOutputCol,
    HasKeepInputData,
    HasImageType,
    HasPageCol,
    DefaultParamsReadable,
    DefaultParamsWritable,
    HasColor,
    HasNumPartitions,
    HasColumnValidator,
    HasDefaultEnum,
    HasPropagateExc,
    metaclass=AutoParamsMeta,
):
    """Crop image by bounding boxes."""

    padding = Param(
        Params._dummy(),
        "padding",
        "Padding.",
        typeConverter=TypeConverters.toInt,
    )

    noCrop = Param(
        Params._dummy(),
        "noCrop",
        "Does not Crop if boxes is empty.",
        typeConverter=TypeConverters.toBoolean,
    )

    limit = Param(
        Params._dummy(),
        "limit",
        "Limit of boxes for crop.",
        typeConverter=TypeConverters.toInt,
    )

    autoRotate = Param(
        Params._dummy(),
        "autoRotate",
        "Auto rotate cropped image if box height > box width.",
        typeConverter=TypeConverters.toBoolean,
    )

    returnEmpty = Param(
        Params._dummy(),
        "returnEmpty",
        "Return Empty list of images in case no boxes.",
        typeConverter=TypeConverters.toBoolean,
    )

    defaultParams = MappingProxyType(
        {
            "inputCols": ["image", "boxes"],
            "outputCol": "cropped_image",
            "keepInputData": False,
            "imageType": ImageType.FILE,
            "numPartitions": 0,
            "padding": 0,
            "pageCol": "page",
            "propagateError": False,
            "noCrop": True,
            "limit": 0,
            "autoRotate": True,
            "returnEmpty": False,
        },
    )

    @keyword_only
    def __init__(self, **kwargs: Any) -> None:
        super(ImageCropBoxes, self).__init__()
        self._setDefault(**self.defaultParams)
        self._set(**kwargs)

    def transform_udf(self, image, data):
        if not isinstance(image, Image):
            image = Image(**image.asDict())
        try:
            if image.exception != "":
                return Image(
                    path=image.path,
                    imageType=image.imageType,
                    data=bytes(),
                    exception=image.exception,
                )
            img = image.to_pil()
            results = []
            limit = self.getLimit()

            bboxes = data.bboxes[:limit] if limit > 0 else data.bboxes

            for b in bboxes:
                box = b
                if not isinstance(box, Box):
                    box = Box(**box.asDict())
                if self.getAutoRotate() and box.width < box.height:
                    cropped_image = img.crop(box.bbox(self.getPadding())).rotate(
                        -90,
                        expand=True,
                    )
                else:
                    cropped_image = img.crop(box.bbox(self.getPadding()))
                results.append(
                    Image.from_pil(
                        cropped_image,
                        image.path,
                        image.imageType,
                        image.resolution,
                    ),
                )

            if self.getNoCrop() and len(results) == 0:
                raise ImageCropError("No boxes to crop")
            if not self.getReturnEmpty() and len(results) == 0:
                results.append(
                    Image.from_pil(img, image.path, image.imageType, image.resolution),
                )

        except Exception as e:
            exception = traceback.format_exc()
            exception = f"ImageCropBoxes: {exception}, {image.exception}"
            logging.warning(exception)
            if self.getPropagateError():
                raise ImageCropError from e
            return [
                Image(image.path, image.imageType, data=bytes(), exception=exception),
            ]
        return results

    def _transform(self, dataset):
        out_col = self.getOutputCol()
        image_col = self._validate(self.getInputCols()[0], dataset)
        box_col = self._validate(self.getInputCols()[1], dataset)

        if self.getNumPartitions() > 0:
            dataset = dataset.repartition(self.getPageCol()).coalesce(
                self.getNumPartitions(),
            )
        result = dataset.withColumn(
            out_col,
            explode(
                udf(self.transform_udf, ArrayType(Image.get_schema()))(
                    image_col,
                    box_col,
                ),
            ),
        )

        if not self.getKeepInputData():
            result = result.drop(image_col)
        return result
