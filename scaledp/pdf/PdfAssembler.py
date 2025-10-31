import logging
import traceback
from types import MappingProxyType
from typing import Any, List

import fitz
import pandas as pd
from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql.functions import udf

from scaledp.params import (
    HasInputCol,
    HasOutputCol,
    Param,
    Params,
    TypeConverters,
)
from scaledp.schemas.PdfDocument import PdfDocument


class HasGroupByCol(Params):
    """
    Mixin for param groupByCol: column name to group by.
    """

    groupByCol: "Param[str]" = Param(
        Params._dummy(),
        "groupByCol",
        "column name to group by.",
        typeConverter=TypeConverters.toString,
    )

    def __init__(self) -> None:
        super(HasGroupByCol, self).__init__()

    def getGroupByCol(self) -> str:
        """
        Gets the value of groupByCol or its default value.
        """
        return self.getOrDefault(self.groupByCol)

    def setGroupByCol(self, value):
        """
        Sets the value of :py:attr:`groupByCol`.
        """
        return self._set(groupByCol=value)


class PdfAssembler(
    Transformer,
    HasInputCol,
    HasOutputCol,
    HasGroupByCol,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    """
    Assembles single-page PDFs into a single PDF document.

    Takes a column containing single-page PDF documents, groups them by origin,
    and creates a single PDF using PyMuPDF (fitz).
    """

    DEFAULT_PARAMS = MappingProxyType(
        {
            "inputCol": "pdf",
            "outputCol": "assembled_pdf",
            "groupByCol": "path",
        },
    )

    @keyword_only
    def __init__(self, **kwargs: Any) -> None:
        super(PdfAssembler, self).__init__()
        self._setDefault(**self.DEFAULT_PARAMS)
        self._set(**kwargs)

    def convert_to_pdf(self, pdfs: List[PdfDocument]) -> PdfDocument:
        """
        Convert a list of single-page PDF documents into a single PDF.

        Args:
            pdfs: List of PdfDocument objects representing single pages

        Returns:
            PdfDocument: A single PDF document containing all pages
        """
        try:
            if not pdfs or len(pdfs) == 0:
                return PdfDocument(
                    path="",
                    data=bytes(),
                    exception="No PDFs to assemble",
                )

            # Filter out invalid PDFs and sort by page if available
            valid_pdfs = []
            for pdf_page in pdfs:
                if (
                    pdf_page.data is not None
                    and len(pdf_page.data) > 0
                    and pdf_page.exception == ""
                ):
                    valid_pdfs.append(pdf_page)

            if not valid_pdfs:
                return PdfDocument(
                    path=pdfs[0].path if pdfs else "",
                    data=bytes(),
                    exception="No valid PDFs to assemble",
                )

            # Create new PDF document
            pdf = fitz.open()

            for pdf_page in valid_pdfs:
                try:
                    # Open the single page PDF
                    page_doc = fitz.open("pdf", pdf_page.data)
                    # Insert all pages from this document (should be just one)
                    pdf.insert_pdf(page_doc)
                    page_doc.close()
                except Exception as e:
                    logging.warning(f"Failed to insert page from {pdf_page.path}: {e}")
                    continue

            # Write the assembled PDF to bytes
            pdf_bytes = pdf.write()
            pdf.close()

            return PdfDocument(
                path=valid_pdfs[0].path,
                data=pdf_bytes,
                exception="",
            )

        except Exception:
            exception = traceback.format_exc()
            exception = f"PdfAssembler: {exception}"
            logging.warning(exception)
            return PdfDocument(
                path=pdfs[0].path if pdfs and len(pdfs) > 0 else "",
                data=bytes(),
                exception=exception,
            )

    def _transform(self, dataset):
        """
        Transform the dataset by grouping single-page PDFs and assembling them.
        """
        output_col = self.getOutputCol()
        input_col = self.getInputCol()
        group_by_col = self.getGroupByCol()

        if input_col not in dataset.columns:
            raise ValueError(
                f"Input column '{input_col}' is not present in the DataFrame.",
            )

        if group_by_col not in dataset.columns:
            raise ValueError(
                f"Group by column '{group_by_col}' is not present in the DataFrame.",
            )

        # Check if we're working with pandas DataFrame (PandasPipeline)
        if isinstance(dataset, pd.DataFrame):
            # Pandas DataFrame approach
            # Sort by path and page number if available
            sort_columns = []
            if "path" in dataset.columns:
                sort_columns.append("path")
            if "page_number" in dataset.columns:
                sort_columns.append("page_number")

            if sort_columns:
                dataset = dataset.sort_values(sort_columns).reset_index(drop=True)

            # Group by the specified column and collect PDFs for each group
            grouped = dataset.groupby(group_by_col)[input_col].apply(list).reset_index()
            grouped.columns = [group_by_col, "pdfs"]

            # Apply the conversion function to each group
            assembled_pdfs = []
            for _, row in grouped.iterrows():
                assembled_pdf = self.convert_to_pdf(row["pdfs"])
                assembled_pdfs.append(assembled_pdf)

            # Create result DataFrame
            result = grouped[[group_by_col]].copy()
            result[output_col] = assembled_pdfs

        else:
            # Spark DataFrame approach (original implementation)
            # Sort by path and page number if available, then group by the specified column
            sorted_dataset = dataset
            if "path" in dataset.columns:
                sorted_dataset = dataset.orderBy("path")
            if "page_number" in dataset.columns:
                sorted_dataset = sorted_dataset.orderBy("page_number")

            # Group by the specified column and collect all PDFs for each group
            from pyspark.sql.functions import collect_list

            grouped_dataset = sorted_dataset.groupBy(dataset[group_by_col]).agg(
                collect_list(dataset[input_col]).alias("pdfs"),
            )

            # Apply the UDF to assemble PDFs for each group
            result_dataset = grouped_dataset.withColumn(
                output_col,
                udf(self.convert_to_pdf, PdfDocument.get_schema())("pdfs"),
            )

            return result_dataset

        return result
