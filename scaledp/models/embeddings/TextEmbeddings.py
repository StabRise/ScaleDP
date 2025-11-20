import json
from dataclasses import asdict
from types import MappingProxyType
from typing import Any

import pandas as pd
from pyspark import keyword_only
from pyspark.sql.types import ArrayType, StructType
from sentence_transformers import SentenceTransformer

from scaledp.enums import Device
from scaledp.models.embeddings.BaseEmbeddings import BaseEmbeddings
from scaledp.schemas.EmbeddingsOutput import EmbeddingsOutput
from scaledp.schemas.TextChunks import TextChunks


class TextEmbeddings(BaseEmbeddings):
    defaultParams = MappingProxyType(
        {
            "inputCol": "text",
            "outputCol": "embeddings",
            "keepInputData": True,
            "model": "all-MiniLM-L6-v2",
            "numPartitions": 1,
            "partitionMap": False,
            "device": Device.CPU,
            "batchSize": 1,
            "pageCol": "page",
            "pathCol": "path",
        },
    )

    @keyword_only
    def __init__(self, **kwargs: Any) -> None:
        super(TextEmbeddings, self).__init__()
        self._setDefault(**self.defaultParams)
        self._set(**kwargs)
        self._model = None

    def _transform(self, dataset):
        """Override _transform to handle both raw text and TextChunks input."""
        from pyspark.sql.functions import explode, lit, pandas_udf, udf

        params = self.get_params()
        out_col = self.getOutputCol()
        input_col = self.getInputCol()

        if input_col not in dataset.columns:
            raise ValueError(f"Column {input_col} not found in dataset")

        # Check if input column is TextChunks (Array type) before validation
        is_chunks = self._is_text_chunks_column(dataset, input_col)

        in_col = self._validate(input_col, dataset)

        if is_chunks:
            # Handle TextChunks input - use explode to create multiple rows
            if not self.getPartitionMap():
                # Use UDF with explode
                result = dataset.withColumn(
                    out_col,
                    explode(
                        udf(
                            self.transform_udf_chunks,
                            ArrayType(EmbeddingsOutput.get_schema()),
                        )(in_col),
                    ),
                )
            else:
                # Use pandas_udf with explode
                if self.getNumPartitions() > 0:
                    if self.getPageCol() in dataset.columns:
                        dataset = dataset.repartition(self.getPageCol())
                    elif self.getPathCol() in dataset.columns:
                        dataset = dataset.repartition(self.getPathCol())
                    dataset = dataset.coalesce(self.getNumPartitions())

                result = dataset.withColumn(
                    out_col,
                    explode(
                        pandas_udf(
                            self.transform_udf_pandas_chunks,
                            ArrayType(EmbeddingsOutput.get_schema()),
                        )(in_col, lit(params)),
                    ),
                )
        elif not self.getPartitionMap():
            result = dataset.withColumn(
                out_col,
                udf(self.transform_udf, EmbeddingsOutput.get_schema())(in_col),
            )
        else:
            if self.getNumPartitions() > 0:
                if self.getPageCol() in dataset.columns:
                    dataset = dataset.repartition(self.getPageCol())
                elif self.getPathCol() in dataset.columns:
                    dataset = dataset.repartition(self.getPathCol())
                dataset = dataset.coalesce(self.getNumPartitions())
            result = dataset.withColumn(
                out_col,
                pandas_udf(
                    self.transform_udf_pandas,
                    EmbeddingsOutput.get_schema(),
                )(
                    in_col,
                    lit(params),
                ),
            )

        if not self.getKeepInputData():
            result = result.drop(in_col)
        return result

    def get_model(self):
        if self._model is None:
            self._model = SentenceTransformer(self.getModel())
        return self._model

    def _is_text_chunks_column(self, dataset, col_name: str) -> bool:
        """Check if the column contains TextChunks data (matches TextChunks schema)."""
        try:
            column_type = dataset.schema[col_name].dataType
            # Check if it's a StructType matching TextChunks schema
            if isinstance(column_type, StructType):
                expected_schema = TextChunks.get_schema()
                return column_type == expected_schema
            return False
        except (KeyError, AttributeError):
            return False

    def transform_udf(self, text: str):
        model = self.get_model()
        embedding = model.encode(
            text,
            batch_size=self.getBatchSize(),
            device=self.getSTDevice(),
        )
        return EmbeddingsOutput(
            path="memory",
            data=embedding.tolist(),
            type="text",
            exception="",
        )

    def transform_udf_chunks(self, text_chunks: TextChunks):
        """Transform TextChunks into embeddings, preserving path information."""
        if not text_chunks or not text_chunks.chunks:
            return []

        model = self.get_model()
        embeddings = model.encode(
            text_chunks.chunks,
            batch_size=self.getBatchSize(),
            device=self.getSTDevice(),
        )
        results = []
        for embedding in embeddings:
            results.append(
                EmbeddingsOutput(
                    path=text_chunks.path or "memory",
                    data=embedding.tolist(),
                    type="text_chunk",
                    exception=text_chunks.exception or "",
                    processing_time=text_chunks.processing_time or 0.0,
                ),
            )
        return results

    @staticmethod
    def transform_udf_pandas(texts: pd.Series, params: pd.Series) -> pd.DataFrame:
        params = json.loads(params[0])
        model = SentenceTransformer(params["model"])
        embeddings = model.encode(
            texts.tolist(),
            batch_size=params["batchSize"],
            device="cpu" if params["device"] == Device.CPU.value else "cuda",
        )
        results = []
        for embedding in embeddings:
            results.append(
                EmbeddingsOutput(
                    path="memory",
                    data=embedding.tolist(),
                    type="text",
                    exception="",
                ),
            )
        return pd.DataFrame(results)

    @staticmethod
    def transform_udf_pandas_chunks(
        chunks_df: pd.DataFrame,
        params: pd.Series,
    ) -> pd.Series:
        """Transform TextChunks into embeddings using pandas_udf, preserving path information."""
        params = json.loads(params.iloc[0])
        model = SentenceTransformer(params["model"])

        results = []
        for _, row in chunks_df.iterrows():
            if len(row["chunks"]):
                embeddings = model.encode(
                    row["chunks"],
                    batch_size=params["batchSize"],
                    device="cpu" if params["device"] == Device.CPU.value else "cuda",
                )
                emb_results = []
                for embedding in embeddings:
                    emb_results.append(
                        asdict(
                            EmbeddingsOutput(
                                path=row.get("path") or "memory",
                                data=embedding.tolist(),
                                type="text_chunk",
                                exception=row.get("exception") or "",
                                processing_time=row.get("processing_time") or 0.0,
                            ),
                        ),
                    )
                results.append(emb_results)
        return pd.Series(results)
