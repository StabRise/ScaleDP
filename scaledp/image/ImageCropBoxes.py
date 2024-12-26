
import traceback
import logging
import random

from PIL import ImageDraw
from pyspark import keyword_only
from pyspark.sql.functions import udf
from pyspark.ml import Transformer
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable

from scaledp.schemas.Box import Box
from scaledp.schemas.Entity import Entity
from scaledp.schemas.Image import Image
from scaledp.schemas.NerOutput import NerOutput
from ..enums import ImageType
from scaledp.params import *


class ImageCropBoxes(Transformer, HasInputCols, HasOutputCol, HasKeepInputData, HasImageType, HasPageCol,
                     DefaultParamsReadable, DefaultParamsWritable, HasColor, HasNumPartitions, HasColumnValidator,
                     HasDefaultEnum, metaclass=AutoParamsMeta):
    """
    Crop image by bounding boces
    """

    padding = Param(Params._dummy(), "padding",
                     "Padding.",
                     typeConverter=TypeConverters.toInt)

    defaultParams = {
        "inputCols": ['image', 'boxes'],
        "outputCol": 'cropped_image',
        "keepInputData": False,
        "imageType": ImageType.FILE,
        "numPartitions": 0,
        "padding": 0,
        "pageCol": "page"
    }

    @keyword_only
    def __init__(self, **kwargs):
        super(ImageCropBoxes, self).__init__()
        self._setDefault(**self.defaultParams)
        self._set(**kwargs)

    def transform_udf(self, image, data):
        if not isinstance(image, Image):
            image = Image(**image.asDict())
        try:
            if image.exception != "":
                return Image(image.origin, image.imageType, data=bytes(), exception=image.exception)
            img = image.to_pil()
            results = []
            for box in data.bboxes:
                if not isinstance(box, Box):
                    box = Box(**box.asDict())
                results.append(img.crop(box.bbox(self.getPadding())))

        except Exception as e:
            exception = traceback.format_exc()
            exception = f"ImageCropBoxes: {exception}, {image.exception}"
            logging.warning(exception)
            return Image(image.path, image.imageType, data=bytes(), exception=exception)
        return Image.from_pil(results[0], image.path, image.imageType, image.resolution)

    def _transform(self, dataset):
        out_col = self.getOutputCol()
        image_col = self._validate(self.getInputCols()[0], dataset)
        box_col = self._validate(self.getInputCols()[1], dataset)

        if self.getNumPartitions() > 0:
            dataset = dataset.repartition(self.getPageCol()).coalesce(self.getNumPartitions())
        result = dataset.withColumn(out_col, udf(self.transform_udf, Image.get_schema())(image_col, box_col))

        if not self.getKeepInputData():
            result = result.drop(image_col)
        return result