import io
import logging
import traceback
from types import MappingProxyType
from typing import Any

import fitz
from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql.functions import udf

from scaledp.params import HasColumnValidator, HasInputCols, HasOutputCol, HasResolution
from scaledp.schemas.PdfDocument import PdfDocument


class PdfAddTextLayer(
    Transformer,
    DefaultParamsReadable,
    DefaultParamsWritable,
    HasOutputCol,
    HasInputCols,
    HasColumnValidator,
    HasResolution,
):
    """Add text layer to PDF document using text from Document schema."""

    DEFAULT_PARAMS = MappingProxyType(
        {
            "inputCols": ["pdf", "text"],
            "outputCol": "pdf_with_text",
            "resolution": 300,  # DPI for coordinate transformation
        },
    )

    @keyword_only
    def __init__(self, **kwargs: Any) -> None:
        super(PdfAddTextLayer, self).__init__()
        self._setDefault(**self.DEFAULT_PARAMS)
        self._set(**kwargs)

    def transform_udf(self, pdf_doc, text_doc):
        """Transform PDF and text documents to create PDF with text layer."""
        try:
            # Check for exceptions in input documents
            if pdf_doc.exception != "":
                return PdfDocument(
                    path=pdf_doc.path,
                    data=bytes(),
                    width=pdf_doc.width,
                    height=pdf_doc.height,
                    exception=pdf_doc.exception,
                )

            if text_doc.exception != "":
                return PdfDocument(
                    path=pdf_doc.path,
                    data=bytes(),
                    width=pdf_doc.width,
                    height=pdf_doc.height,
                    exception=text_doc.exception,
                )

            if not pdf_doc.data:
                return PdfDocument(
                    path=pdf_doc.path,
                    data=bytes(),
                    width=pdf_doc.width,
                    height=pdf_doc.height,
                    exception="PDF document has no data",
                )

            # Open the PDF document
            pdf_document = fitz.open(stream=pdf_doc.data, filetype="pdf")

            if len(pdf_document) == 0:
                pdf_document.close()
                return PdfDocument(
                    path=pdf_doc.path,
                    data=bytes(),
                    width=pdf_doc.width,
                    height=pdf_doc.height,
                    exception="PDF document has no pages",
                )

            # Get the first page (assuming single page PDF as per requirement)
            page = pdf_document[0]

            # Calculate scale factor from image coordinates to PDF coordinates
            pdf_dpi = 72.0  # PDF native DPI
            scale_factor = pdf_dpi / self.getResolution()

            # Add text layer using bounding boxes from Document
            if text_doc.bboxes:
                for bbox in text_doc.bboxes:
                    # Convert image coordinates to PDF coordinates
                    # Image coordinates: origin at top-left, y increases downward
                    # PDF coordinates: origin at bottom-left, y increases upward

                    pdf_x = bbox.x * scale_factor
                    pdf_y = (bbox.y - 0.2 * bbox.height) * scale_factor  # Flip Y axis
                    pdf_height = bbox.height * scale_factor

                    # Insert text at the specified position using the correct PyMuPDF method
                    page.insert_text(
                        point=fitz.Point(
                            pdf_x,
                            pdf_y + pdf_height,
                        ),  # Bottom-left of text
                        text=bbox.text,
                        fontsize=max(8, pdf_height * 0.8),  # Scale font size
                        # with bbox height
                        color=(0, 0, 0),  # Black text
                        overlay=False,
                    )

            # Save the modified PDF to bytes
            output_buffer = io.BytesIO()
            pdf_document.save(output_buffer)
            pdf_bytes = output_buffer.getvalue()
            pdf_document.close()

            return PdfDocument(
                path=pdf_doc.path,
                data=pdf_bytes,
                width=pdf_doc.width,
                height=pdf_doc.height,
                exception="",
            )

        except Exception:
            exception = traceback.format_exc()
            exception = f"PdfAddTextLayer: {exception}"
            logging.warning(exception)
            return PdfDocument(
                path=pdf_doc.path if pdf_doc else "",
                data=bytes(),
                width=pdf_doc.width if pdf_doc else None,
                height=pdf_doc.height if pdf_doc else None,
                exception=exception,
            )

    def _transform(self, dataset):
        """Transform the dataset by adding text layer to PDF documents."""
        output_col = self.getOutputCol()
        input_cols = self.getInputCols()

        # Validate that we have exactly 2 input columns
        if len(input_cols) != 2:
            raise ValueError(
                f"PdfAddTextLayer requires exactly 2 input columns "
                f"(PDF and text), got {len(input_cols)}",
            )

        pdf_col, text_col = input_cols

        # Validate input columns exist
        if pdf_col not in dataset.columns:
            raise ValueError(
                f"PDF input column '{pdf_col}' is not present in the DataFrame.",
            )

        if text_col not in dataset.columns:
            raise ValueError(
                f"Text input column '{text_col}' is not present in the DataFrame.",
            )

        pdf_column = dataset[pdf_col]
        text_column = dataset[text_col]

        return dataset.withColumn(
            output_col,
            udf(self.transform_udf, PdfDocument.get_schema())(pdf_column, text_column),
        )
