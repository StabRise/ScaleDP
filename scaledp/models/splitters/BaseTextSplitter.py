import json
from abc import abstractmethod
from types import MappingProxyType
from typing import Any

import pandas as pd
from pyspark import keyword_only
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql.functions import lit, pandas_udf, udf

from scaledp.params import HasColumnValidator, HasDefaultEnum
from scaledp.schemas.Document import Document
from scaledp.schemas.TextChunks import TextChunks

from .BaseSplitter import BaseSplitter


class BaseTextSplitter(
    BaseSplitter,
    DefaultParamsReadable,
    DefaultParamsWritable,
    HasColumnValidator,
    HasDefaultEnum,
):
    """
    Abstract base class for text splitters that split text into chunks.
    Provides common functionality for text splitting operations.
    """

    defaultParams = MappingProxyType(
        {
            "inputCol": "document",
            "outputCol": "chunks",
            "keepInputData": True,
            "chunk_size": 500,
            "chunk_overlap": 0,
            "numPartitions": 1,
            "partitionMap": False,
            "pageCol": "page_number",
        },
    )

    @keyword_only
    def __init__(self, **kwargs: Any):
        super(BaseTextSplitter, self).__init__()
        self._setDefault(**self.defaultParams)
        self._set(**kwargs)

    def get_params(self):
        """Get transformer parameters as JSON string."""
        return json.dumps({k.name: v for k, v in self.extractParamMap().items()})

    @abstractmethod
    def split(self, document: Document, pagen_number: int) -> TextChunks:
        """
        Split a document into chunks.

        Args:
            document: The document to split
            pagen_number: The page number of the document

        Returns:
            TextChunks object containing the chunks and metadata
        """

    def transform_udf(self, document_struct, page_number):
        """
        Transform UDF that splits text into chunks.

        Args:
            document_struct: A Document struct containing path, text, type, and bboxes

        Returns:
            TextChunks object containing the chunks
        """
        # document_struct is already a Document object
        result = self.split(document_struct, page_number)
        return result

    @classmethod
    def transform_udf_pandas(
        cls,
        documents: pd.DataFrame,
        page_numbers: pd.Series,
        params: pd.Series,
    ) -> pd.DataFrame:
        """
        Transform pandas Series of Documents using the splitter.

        Args:
            documents: Series containing Document objects (as Row-like objects from Arrow)
            params: Series containing splitter parameters

        Returns:
            DataFrame with TextChunks results
        """
        params_dict = json.loads(params.iloc[0])
        splitter = cls(**params_dict)
        results = []
        for i, doc_row in documents.iterrows():
            # Convert Row to Document
            # When using pandas_udf with Arrow, the struct comes as
            # a Row object with field attributes
            try:
                doc = (
                    doc_row
                    if isinstance(doc_row, Document)
                    else Document(**doc_row.to_dict())
                )
                output = splitter.split(doc, page_numbers.iloc[i])
            except (AttributeError, TypeError, Exception) as e:
                # If something goes wrong, create an error result
                output = TextChunks(
                    path="",
                    chunks=[],
                    exception=str(e),
                    processing_time=0.0,
                )
            # Convert to dict to ensure proper schema
            results.append(output)
        return pd.DataFrame(results)

    def _transform(self, dataset):
        """
        Transform a DataFrame by splitting text into chunks.

        Args:
            dataset: Input DataFrame with Document struct column

        Returns:
            DataFrame with chunks column added
        """
        params = self.get_params()
        out_col = self.getOutputCol()
        input_col = self.getInputCol()
        page_col = self.getPageCol()

        # Validate input column exists
        if input_col not in dataset.columns:
            raise ValueError(f"Column {input_col} not found in dataset")

        # Validate input column
        validated_input_col = self._validate(input_col, dataset)
        validated_page_col = self._validate(page_col, dataset)

        if not self.getPartitionMap():
            # Regular mode: use UDF
            result = dataset.withColumn(
                out_col,
                udf(self.transform_udf, TextChunks.get_schema())(
                    validated_input_col,
                    validated_page_col,
                ),
            )
        else:
            # Pandas mode: use pandas_udf
            if self.getNumPartitions() > 0:
                dataset = dataset.coalesce(self.getNumPartitions())

            result = dataset.withColumn(
                out_col,
                pandas_udf(self.transform_udf_pandas, TextChunks.get_schema())(
                    validated_input_col,
                    validated_page_col,
                    lit(params),
                ),
            )

        if not self.getKeepInputData():
            result = result.drop(validated_input_col)

        return result
