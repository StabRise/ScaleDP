import logging
import traceback
from types import MappingProxyType
from typing import Any

import fitz
from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.pandas import DataFrame
from pyspark.sql.functions import posexplode_outer, udf
from pyspark.sql.types import ArrayType, Row

from scaledp.params import (
    HasColumnValidator,
    HasInputCol,
    HasKeepInputData,
    HasOutputCol,
    HasPageCol,
    HasPathCol,
)
from scaledp.schemas.Box import Box
from scaledp.schemas.Document import Document


class PdfDataToText(
    Transformer,
    HasInputCol,
    HasOutputCol,
    HasKeepInputData,
    HasPathCol,
    HasPageCol,
    DefaultParamsReadable,
    DefaultParamsWritable,
    HasColumnValidator,
):
    """Extract text with coordinates from PDF file."""

    DEFAULT_PARAMS = MappingProxyType(
        {
            "inputCol": "content",
            "outputCol": "document",
            "pathCol": "path",
            "pageCol": "page",
            "keepInputData": False,
        },
    )

    @keyword_only
    def __init__(self, **kwargs: Any) -> None:
        super(PdfDataToText, self).__init__()
        self._setDefault(**self.DEFAULT_PARAMS)
        self._set(**kwargs)

    def transform_udf(self, input: Row, path: str) -> list[Document]:
        logging.info("Run Pdf Data to Text")
        try:
            doc = fitz.open("pdf", input)
            if len(doc) == 0:
                raise ValueError("Empty PDF document.")

            result = []
            for page in doc:
                words = page.get_text("words")
                boxes = []
                text_content = []

                for word in words:
                    x0, y0, x1, y1, word_text, _, _, _ = word
                    boxes.append(
                        Box(
                            x=int(x0),
                            y=int(y0),
                            width=int(x1 - x0),
                            height=int(y1 - y0),
                            text=word_text,
                            score=1.0,
                        ),
                    )
                    text_content.append(word_text)

                result.append(
                    Document(
                        path=path,
                        text=" ".join(text_content),
                        type="pdf",
                        bboxes=boxes,
                    ),
                )
            return result

        except Exception:
            exception = traceback.format_exc()
            exception = (
                f"{self.uid}: Error during extracting text from "
                f"the PDF document: {exception}"
            )
            logging.warning(exception)
            return [
                Document(
                    path=path,
                    text="",
                    type="pdf",
                    bboxes=[],
                    exception=exception,
                ),
            ]

    def _transform(self, dataset: DataFrame) -> DataFrame:
        out_col = self.getOutputCol()
        input_col = self._validate(self.getInputCol(), dataset)
        path_col = dataset[self.getPathCol()]

        sel_col = [
            *dataset.columns,
            *[
                posexplode_outer(
                    udf(self.transform_udf, ArrayType(Document.get_schema()))(
                        input_col,
                        path_col,
                    ),
                ).alias(self.getPageCol(), out_col),
            ],
        ]

        result = dataset.select(*sel_col)
        if not self.getKeepInputData():
            result = result.drop(input_col)
        return result
