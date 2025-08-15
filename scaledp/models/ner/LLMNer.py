import json
from types import MappingProxyType
from typing import Any, List

import pandas as pd
from pydantic import BaseModel
from pyspark import keyword_only
from pyspark.ml.param import Param, Params, TypeConverters
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from scaledp.models.ner.BaseNer import BaseNer
from scaledp.params import HasLLM, HasPrompt, HasPropagateExc
from scaledp.schemas.Entity import Entity
from scaledp.schemas.NerOutput import NerOutput

from ...enums import Device


class LLMNer(BaseNer, HasLLM, HasPrompt, HasPropagateExc):

    tags = Param(
        Params._dummy(),
        "tags",
        "Ner tags.",
        typeConverter=TypeConverters.toListString,
    )

    defaultParams = MappingProxyType(
        {
            "inputCols": ["text"],
            "outputCol": "ner",
            "keepInputData": True,
            "whiteList": [],
            "numPartitions": 1,
            "device": Device.CPU,
            "batchSize": 1,
            "scoreThreshold": 0.0,
            "pageCol": "page",
            "pathCol": "path",
            "systemPrompt": "You are excellent NER tag extractor.",
            "prompt": """Please extract text from the image.""",
            "model": "gemini-1.5-flash-8b",
            "apiBase": "",
            "apiKey": "",
            "delay": 30,
            "maxRetry": 6,
            "propagateError": False,
            "tags": [
                "PERSON",
                "LOCATION",
                "DATE",
                "EMAIL",
                "PHONE",
                "ORGANIZATION",
                "ID",
            ],
            "partitionMap": False,
        },
    )

    def getPaydanticSchema(self):

        class Entity(BaseModel):
            entity_group: str
            word: str

        class NerOutput(BaseModel):
            entities: List[Entity]

        return NerOutput

    @keyword_only
    def __init__(self, **kwargs: Any) -> None:
        super(LLMNer, self).__init__()
        self._setDefault(**self.defaultParams)
        self._set(**kwargs)
        self.pipeline = None

    def transform_udf(self, document, params=None):

        if params is None:
            params = self.get_params()
        params = json.loads(params)

        mapping = []
        for idx, box in enumerate(document.bboxes):
            mapping.extend([idx] * (len(box.text) + 1))

        from openai import RateLimitError

        client = self.getClient(params["apiKey"], params["apiBase"])

        @retry(
            retry=retry_if_exception_type(RateLimitError),
            wait=wait_random_exponential(min=1, max=params["delay"]),
            stop=stop_after_attempt(params["maxRetry"]),
        )
        def completion_with_backoff(**kwargs: Any):
            return client.beta.chat.completions.parse(**kwargs)

        schema = self.getPaydanticSchema().model_json_schema()

        completion = completion_with_backoff(
            model=params["model"],
            messages=[
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": params["systemPrompt"]},
                    ],
                },
                {
                    "role": "user",
                    "content": f'Pleas extract NER tags: {",".join(params["tags"])}'
                    f" as json with schema: {schema}. "
                    f"Return only valid json without any other extra text. From the text:"
                    + document.text,
                },
            ],
            response_format={"type": "json_object"},
        )
        entities = []

        data = json.loads(
            completion.choices[0]
            .message.content.replace("```json", "")
            .replace("```", ""),
        )
        for tag in data["entities"]:
            if (
                len(self.getWhiteList()) > 0
                and tag["entity_group"] not in self.getWhiteList()
            ):
                continue
            boxes = []
            word = tag["word"]
            for _idx, box in enumerate(document.bboxes):
                if any(
                    word.lower() in box.text.lower()
                    and (len(word) > 2 or abs(len(word) - len(box.text)) < 2)
                    for word in word.replace(",", " ").replace("@", " ").split(" ")
                    if len(word) > 1
                ):
                    boxes.append(box)
            t = Entity(
                entity_group=tag["entity_group"],
                score=0,  # float(tag["score"]),
                word=tag["word"],
                start=0,  # tag["start"],
                end=0,  # tag["end"],
                boxes=boxes,
            )
            entities.append(t)

        return NerOutput(path=document.path, entities=entities, exception="")

    @staticmethod
    def transform_udf_pandas(
        texts: pd.DataFrame,
        params: pd.Series,
    ) -> pd.DataFrame:  # pragma: no cover
        results = []

        return pd.DataFrame(results)
