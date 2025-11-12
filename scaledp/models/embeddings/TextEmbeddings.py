import json
from types import MappingProxyType
from typing import Any

import pandas as pd
from pyspark import keyword_only
from sentence_transformers import SentenceTransformer

from scaledp.enums import Device
from scaledp.models.embeddings.BaseEmbeddings import BaseEmbeddings
from scaledp.schemas.EmbeddingsOutput import EmbeddingsOutput


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

    def get_model(self):
        if self._model is None:
            self._model = SentenceTransformer(self.getModel())
        return self._model

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
