import logging
import math
import random
import traceback
from types import MappingProxyType
from typing import Any

from PIL import ImageDraw
from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql.functions import udf

from scaledp.params import (
    AutoParamsMeta,
    HasBlackList,
    HasColor,
    HasColumnValidator,
    HasDefaultEnum,
    HasImageType,
    HasInputCols,
    HasKeepInputData,
    HasNumPartitions,
    HasOutputCol,
    HasPageCol,
    HasWhiteList,
)
from scaledp.schemas.Box import Box
from scaledp.schemas.Entity import Entity
from scaledp.schemas.Image import Image
from scaledp.schemas.NerOutput import NerOutput

from ..enums import ImageType


class ImageDrawBoxes(
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
    HasWhiteList,
    HasBlackList,
    metaclass=AutoParamsMeta,
):
    """Draw boxes on image."""

    filled = Param(
        Params._dummy(),
        "filled",
        "Fill rectangle.",
        typeConverter=TypeConverters.toBoolean,
    )

    lineWidth = Param(
        Params._dummy(),
        "lineWidth",
        "Line width.",
        typeConverter=TypeConverters.toInt,
    )

    textSize = Param(
        Params._dummy(),
        "textSize",
        "Text size.",
        typeConverter=TypeConverters.toInt,
    )

    padding = Param(
        Params._dummy(),
        "padding",
        "Padding.",
        typeConverter=TypeConverters.toInt,
    )

    displayDataList = Param(
        Params._dummy(),
        "displayDataList",
        "Display data list.",
        typeConverter=TypeConverters.toListString,
    )

    defaultParams = MappingProxyType(
        {
            "inputCols": ["image", "boxes"],
            "outputCol": "image_with_boxes",
            "keepInputData": False,
            "imageType": ImageType.FILE,
            "filled": False,
            "color": None,
            "lineWidth": 1,
            "textSize": 12,
            "displayDataList": [],
            "numPartitions": 0,
            "padding": 0,
            "pageCol": "page",
            "whiteList": [],
            "blackList": [],
        },
    )

    @keyword_only
    def __init__(self, **kwargs: Any) -> None:
        super(ImageDrawBoxes, self).__init__()
        self._setDefault(**self.defaultParams)
        self._set(**kwargs)

    def getDisplayText(self, box):
        display_list = self.getDisplayDataList()
        text = []
        if display_list:
            for name in display_list:
                if hasattr(box, name):
                    val = getattr(box, name)
                    if isinstance(val, float):
                        text.append(f"{val:0.2f}")
                    elif isinstance(val, int):
                        text.append(str(val))
                    else:
                        text.append(val)
        return ":".join(text)

    def transform_udf(self, image, *data_list: Any):

        def get_color():
            return "#{:06x}".format(random.randint(0, 0xFFFFFF))

        if not isinstance(image, Image):
            image = Image(**image.asDict())
        try:
            if image.exception != "":
                return Image(
                    image.origin,
                    image.imageType,
                    data=bytes(),
                    exception=image.exception,
                )
            img = image.to_pil()
            img1 = ImageDraw.Draw(img)
            fill = self.getColor() if self.getFilled() else None

            for data in data_list:

                if hasattr(data, "entities"):
                    self.draw_ner_boxes(data, fill, get_color, img1)
                else:
                    self.draw_boxes(data, fill, img1)

        except Exception:
            exception = traceback.format_exc()
            exception = f"ImageDrawBoxes: {exception}, {image.exception}"
            logging.warning(exception)
            return Image(image.path, image.imageType, data=bytes(), exception=exception)
        return Image.from_pil(img, image.path, image.imageType, image.resolution)

    def draw_boxes(self, data, fill, img1):
        colors = {}
        for b in data.bboxes:
            box = b
            if not isinstance(box, Box):
                box = Box(**box.asDict())

            # Group by Box.text field for color consistency
            text_key = box.text if hasattr(box, "text") and box.text else "default"

            if text_key not in colors:
                colors[text_key] = "#{:06x}".format(random.randint(0, 0xFFFFFF))

            color = colors[text_key] if self.getColor() is None else self.getColor()

            self.draw_box(box, color, fill, img1)
            text = self.getDisplayText(box)
            if text:
                tbox = list(
                    img1.textbbox(
                        (
                            box.x,
                            box.y - self.getTextSize() * 1.2 - self.getPadding(),
                        ),
                        text,
                        font_size=self.getTextSize(),
                    ),
                )
                tbox[3] = tbox[3] + self.getTextSize() / 4
                tbox[2] = tbox[2] + self.getTextSize() / 4
                tbox[0] = box.x - self.getPadding()
                tbox[1] = box.y - self.getTextSize() * 1.2 - self.getPadding()
                img1.rounded_rectangle(
                    tbox,
                    outline=color,
                    radius=2,
                    fill=color,
                )
                img1.text(
                    (
                        box.x,
                        box.y - self.getTextSize() * 1.2 - self.getPadding(),
                    ),
                    text,
                    fill="white",
                    font_size=self.getTextSize(),
                )

    def draw_box(self, box, color, fill, img1):
        if box.angle == 0:
            # Draw normal rectangle if angle is 0
            img1.rounded_rectangle(
                box.shape(self.getPadding()),
                outline=color,
                radius=4,
                fill=fill,
                width=self.getLineWidth(),
            )
        else:
            # Draw rotated rectangle for non-zero angles
            center_x = box.x + box.width / 2
            center_y = box.y + box.height / 2
            points = [
                # Top-left
                (-box.width / 2, -box.height / 2),
                # Top-right
                (box.width / 2, -box.height / 2),
                # Bottom-right
                (box.width / 2, box.height / 2),
                # Bottom-left
                (-box.width / 2, box.height / 2),
            ]
            # Rotate points and translate to center
            rotated_points = []

            angle_rad = math.radians(box.angle)
            for px, py in points:
                # Rotate point
                rx = px * math.cos(angle_rad) - py * math.sin(angle_rad)
                ry = px * math.sin(angle_rad) + py * math.cos(angle_rad)
                # Translate to center
                rx += center_x
                ry += center_y
                rotated_points.append((rx, ry))

            # Draw polygon with rotated points
            img1.polygon(
                rotated_points,
                outline=color,
                fill=fill,
                width=self.getLineWidth(),
            )

    def draw_ner_boxes(self, data, fill, get_color, img1):
        black_list = self.getBlackList()
        white_list = self.getWhiteList()
        if not isinstance(data, NerOutput):
            data = NerOutput(**data.asDict())
        colors = {}
        for n in data.entities:
            ner = n
            if not isinstance(ner, Entity):
                ner = Entity(**ner.asDict())

            if ner.entity_group in black_list:
                continue
            if white_list and ner.entity_group not in white_list:
                continue

            if ner.entity_group not in colors:
                colors[ner.entity_group] = get_color()
            color = (
                colors[ner.entity_group] if self.getColor() is None else self.getColor()
            )
            for b in ner.boxes:
                box = b
                if not isinstance(box, Box):
                    box = Box(**box.asDict())
                text = self.getDisplayText(ner)
                self.draw_box(box, color, fill, img1)

                if text:
                    tbox = list(
                        img1.textbbox(
                            (
                                box.x,
                                box.y - self.getTextSize() * 1.2 - self.getPadding(),
                            ),
                            text,
                            font_size=self.getTextSize(),
                        ),
                    )
                    tbox[3] = tbox[3] + self.getTextSize() / 4
                    tbox[2] = tbox[2] + self.getTextSize() / 4
                    tbox[0] = box.x - self.getPadding()
                    tbox[1] = box.y - self.getTextSize() * 1.2 - self.getPadding()
                    img1.rounded_rectangle(
                        tbox,
                        outline=color,
                        radius=2,
                        fill=color,
                    )
                    img1.text(
                        (
                            box.x,
                            box.y - self.getTextSize() * 1.2 - self.getPadding(),
                        ),
                        text,
                        stroke_width=0,
                        fill="white",
                        font_size=self.getTextSize(),
                    )

    def _preprocessing(self, dataset):
        return dataset

    def _transform(self, dataset):
        out_col = self.getOutputCol()
        image_col = self._validate(self.getInputCols()[0], dataset)
        box_cols = [self._validate(col, dataset) for col in self.getInputCols()[1:]]

        dataset = self._preprocessing(dataset)

        if self.getNumPartitions() > 0:
            dataset = dataset.repartition(self.getPageCol()).coalesce(
                self.getNumPartitions(),
            )
        result = dataset.withColumn(
            out_col,
            udf(self.transform_udf, Image.get_schema())(image_col, *box_cols),
        )

        if not self.getKeepInputData():
            result = result.drop(image_col)
        return result
