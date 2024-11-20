import traceback
import logging
import json

import pandas as pd
from pyspark.sql.types import *
from scaledp.params import *
from pyspark.ml import Transformer
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable

from ...utils import get_size, cluster
from pyspark.sql.functions import udf, pandas_udf, lit
from scaledp.schemas.Document import Document
from scaledp.schemas.Image import Image


class BaseOcr(Transformer, HasInputCol, HasOutputCol, HasKeepInputData, HasDefaultEnum,
              DefaultParamsReadable, DefaultParamsWritable, HasScoreThreshold, HasLang,
              HasPartitionMap, HasColumnValidator, HasNumPartitions, HasPageCol, HasPathCol):

    scaleFactor = Param(Params._dummy(), "scaleFactor",
                      "Scale Factor.",
                      typeConverter=TypeConverters.toFloat)

    keepFormatting = Param(Params._dummy(), "keepFormatting",
                           "Whether to keep the original formatting.",
                           typeConverter=TypeConverters.toBoolean)

    lineTolerance = Param(Params._dummy(), "lineTolerance",
                          "Tolerance for line clustering.",
                          typeConverter=TypeConverters.toInt)

    @staticmethod
    def to_formatted_text(lines, character_height):
        output_lines = []
        space_width = BaseOcr.get_character_width(lines)
        y = 0
        for regions in lines:
            line = ""
            # Add extra empty lines if need
            line_diffs = int((regions[0].y - y) / (character_height * 2))
            y = regions[0].y
            if line_diffs > 1:
                for i in range(line_diffs - 1):
                    output_lines.append("")

            prev = 0
            for region in regions:
                left2 = region.x - prev
                spaces = max(int(left2 / space_width), 1)
                line = line + spaces * " " + region.text
                prev = region.x + region.width
            output_lines.append(line)
        return "\n".join(output_lines)

    @staticmethod
    def get_character_width(lines):
        character_widths = []
        for regions in lines:
            for region in regions:
                width = region.width
                character_widths.append(int(width / len(region.text)))
        return get_size(character_widths)

    @staticmethod
    def box_to_formatted_text(boxes, line_tolerance=0):
        character_height = get_size(boxes, lambda x: x.height)
        line_tolerance = character_height / 3
        if line_tolerance != 0:
            line_tolerance = line_tolerance

        lines = cluster(boxes, line_tolerance, key=lambda i: int(i.y))

        lines = [
            sorted(xs, key=lambda i: int(i.x))
            for xs in lines
        ]
        return BaseOcr.to_formatted_text(lines, character_height)

    def transform_udf(self, image, params=None):
        logging.info("Run OCR")
        if params is None:
            params = self.get_params()
        params = json.loads(params)
        if not isinstance(image, Image):
            image = Image(**image.asDict())
        if image.exception != "":
            return Document(path=image.path,
                            text="",
                            bboxes=[],
                            type="text",
                            exception=image.exception)
        try:
            image_pil = image.to_pil()
            scale_factor = self.getScaleFactor()
            if scale_factor != 1.0:
                resized_image = image_pil.resize((int(image_pil.width * scale_factor), int(image_pil.height * scale_factor)))
            else:
                resized_image = image_pil

            result = self.call_ocr([(resized_image, image.path)], params)
        except Exception as e:
            exception = traceback.format_exc()
            exception = f"{self.uid}: Error in text recognition: {exception}, {image.exception}"
            logging.warning(f"{self.uid}: Error in text recognition.")
            return Document(path=image.path,
                            text="",
                            bboxes=[],
                            type="ocr",
                            exception=exception)
        return result[0]

    @classmethod
    def call_ocr(cls, resized_images,  params):
        raise NotImplementedError("Subclasses should implement this method")

    @classmethod
    def transform_udf_pandas(cls, images: pd.DataFrame, params: pd.Series) -> pd.DataFrame:
        params = json.loads(params[0])

        resized_images = []
        for index, image in images.iterrows():
            if not isinstance(image, Image):
                image = Image(**image.to_dict())
            image_pil = image.to_pil()
            scale_factor = params['scaleFactor']
            if scale_factor != 1.0:
                resized_image = image_pil.resize(
                    (int(image_pil.width * scale_factor), int(image_pil.height * scale_factor)))
            else:
                resized_image = image_pil
            resized_images.append((resized_image, image.path))

        results = cls.call_ocr(resized_images, params)

        return pd.DataFrame(results)


    def outputSchema(self):
        return StructType([StructField("path", StringType(), True),
                           StructField("text", StringType(), True),
                           StructField("type", StringType(), True),
                           StructField("bboxes", ArrayType(StructType(
                                             [StructField("text", StringType(), False)
                                                 , StructField("score", DoubleType(), False),
                                              StructField("x", IntegerType(), False),
                                              StructField("y", IntegerType(), False),
                                              StructField("width", IntegerType(), False),
                                              StructField("height", IntegerType(), False)]),
                                                                        True),
                                                     True),
                           StructField("exception", StringType(), True)])

    def get_params(self):
        return json.dumps({k.name: v for k, v in self.extractParamMap().items()})

    def _transform(self, dataset):
        out_col = self.getOutputCol()
        input_col = self._validate(self.getInputCol(), dataset)
        params = self.get_params()
        if not self.getPartitionMap():
            result = dataset.withColumn(out_col, udf(self.transform_udf, Document.get_schema())(input_col, lit(params)))
        else:
            if self.getNumPartitions() > 0:
                if self.getPageCol() in dataset.columns:
                    dataset = dataset.repartition( self.getPageCol())
                elif self.getPathCol() in dataset.columns:
                    dataset = dataset.repartition(self.getPathCol())
                dataset = dataset.coalesce(self.getNumPartitions())
            result = dataset.withColumn(out_col,
                                        pandas_udf(self.transform_udf_pandas, self.outputSchema())
                                        (input_col, lit(params)))
        if not self.getKeepInputData():
            result = result.drop(input_col)
        return result

    def setLineTolerance(self, value):
        """
        Sets the value of :py:attr:`lineTolerance`.
        """
        return self._set(lineTolerance=value)

    def getLineTolerance(self):
        """
        Gets the value of :py:attr:`lineTolerance`.
        """
        return self.getOrDefault(self.lineTolerance)

    def setKeepFormatting(self, value):
        """
        Sets the value of :py:attr:`keepFormatting`.
        """
        return self._set(keepFormatting=value)

    def getKeepFormatting(self):
        """
        Gets the value of :py:attr:`keepFormatting`.
        """
        return self.getOrDefault(self.keepFormatting)

    def setScaleFactor(self, value):
        """
        Sets the value of :py:attr:`scaleFactor`.
        """
        return self._set(scaleFactor=value)

    def getScaleFactor(self):
        """
        Sets the value of :py:attr:`scaleFactor`.
        """
        return self.getOrDefault(self.scaleFactor)