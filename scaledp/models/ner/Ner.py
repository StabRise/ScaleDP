import json
from types import MappingProxyType
from typing import Any

import pandas as pd
from pyspark import keyword_only

from scaledp.models.ner.BaseNer import BaseNer
from scaledp.schemas.Document import Document
from scaledp.schemas.Entity import Entity
from scaledp.schemas.NerOutput import NerOutput

from ...enums import Device


class Ner(BaseNer):

    defaultParams = MappingProxyType(
        {
            "inputCols": ["text"],
            "outputCol": "ner",
            "keepInputData": True,
            "model": "d4data/biomedical-ner-all",
            "whiteList": [],
            "numPartitions": 1,
            "partitionMap": False,
            "device": Device.CPU,
            "batchSize": 1,
            "scoreThreshold": 0.0,
            "pageCol": "page",
            "pathCol": "path",
        },
    )

    @keyword_only
    def __init__(self, **kwargs: Any) -> None:
        super(Ner, self).__init__()
        self._setDefault(**self.defaultParams)
        self._set(**kwargs)
        self.pipeline = None

    @staticmethod
    def split_text(text, max_length=500, stride=256):
        chunks = []
        for i in range(0, len(text), stride):
            chunk = text[i : i + max_length]
            chunks.append(chunk)
            if len(chunk) < max_length:
                break
        return chunks

    @staticmethod
    def aggregate_ner_results(text, pipeline, max_length=500, stride=256):
        chunks = Ner.split_text(text, max_length=max_length, stride=stride)
        entities = []
        for num, chunk in enumerate(chunks):
            ner_results = pipeline(chunk)
            new_ner_results = []
            for i in ner_results:
                i["start"] = i["start"] + num * stride
                i["end"] = i["end"] + num * stride
                new_ner_results.append(i)
            entities.extend(ner_results)
        return entities

    def get_pipeline(self):
        if self.pipeline is None:
            from transformers import (
                AutoModelForTokenClassification,
                AutoTokenizer,
                pipeline,
            )

            tokenizer = AutoTokenizer.from_pretrained(self.getModel())
            ort_model = AutoModelForTokenClassification.from_pretrained(self.getModel())
            self.pipeline = pipeline(
                "ner",
                model=ort_model,
                tokenizer=tokenizer,
                aggregation_strategy="first",
                device=int(self.getDevice()),
            )
        return self.pipeline

    def transform_udf(self, *documents: list[Document]):

        mapping = []
        typed_document = []
        if not isinstance(documents[0], Document):
            for document in documents:
                typed_document.append(Document(**document.asDict()))
        else:
            typed_document = documents
        text = typed_document[0]
        for t in typed_document[1:]:
            text = text.merge(t)

        for idx, box in enumerate(text.bboxes):
            mapping.extend([idx] * (len(box.text) + 1))

        result = Ner.aggregate_ner_results(
            text.text,
            self.get_pipeline(),
            max_length=500,
            stride=480,
        )

        entities = []
        for tag in result:
            if (
                len(self.getWhiteList()) > 0
                and tag["entity_group"] not in self.getWhiteList()
            ):
                continue
            boxes = mapping[tag["start"] : tag["end"]]
            boxes = [text.bboxes[i] for i in list(dict.fromkeys(boxes))]
            t = Entity(
                entity_group=tag["entity_group"],
                score=float(tag["score"]),
                word=tag["word"],
                start=tag["start"],
                end=tag["end"],
                boxes=boxes,
            )
            entities.append(t)
        return NerOutput(path=text.path, entities=entities, exception="")

    @staticmethod
    def transform_udf_pandas(
        texts: pd.DataFrame,
        params: pd.Series,
    ) -> pd.DataFrame:  # pragma: no cover
        params = json.loads(params[0])
        model = params["model"]
        from transformers import (
            AutoModelForTokenClassification,
            AutoTokenizer,
            pipeline,
        )

        tokenizer = AutoTokenizer.from_pretrained(model)
        model = AutoModelForTokenClassification.from_pretrained(model)
        pipe = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="first",
            device=int(params["device"]),
        )

        results = []
        max_length = 500
        stride = 480
        batch_texts = [
            (index, Ner.split_text(text["text"] or "-", max_length, stride))
            for index, text in texts.iterrows()
        ]

        if all(all(chunk == "-" for chunk in chunks) for index, chunks in batch_texts):
            for _index, text in texts.iterrows():
                output = NerOutput(path=text.path, entities=[], exception="")
                results.append(output)
            return pd.DataFrame(results)

        chunks = []
        offsets = []
        for index, exit_chunks in batch_texts:
            for num, chunk in enumerate(exit_chunks):
                chunks.append(chunk)
                offsets.append((index, num * stride))

        ner_results = pipe(chunks)
        batch_results = [[] for i in range(len(batch_texts))]
        for num, ner_result in enumerate(ner_results):

            for i in ner_result:
                i["start"] = i["start"] + offsets[num][1]
                i["end"] = i["end"] + offsets[num][1]
                batch_results[offsets[num][0]].append(i)

        for (_index, text), result in zip(texts.iterrows(), batch_results):
            mapping = []
            for idx, box in enumerate(text["bboxes"]):
                mapping.extend([idx] * (len(box["text"]) + 1))

            entities = []
            for tag in result:
                if (
                    len(params["whiteList"]) > 0
                    and tag["entity_group"] not in params["whiteList"]
                ):
                    continue
                if (
                    params["scoreThreshold"] > 0
                    and tag["score"] < params["scoreThreshold"]
                ):
                    continue
                boxes = mapping[tag["start"] : tag["end"]]
                boxes = [text.bboxes[i] for i in list(dict.fromkeys(boxes))]
                t = Entity(
                    entity_group=tag["entity_group"],
                    score=float(tag["score"]),
                    word=tag["word"],
                    start=tag["start"],
                    end=tag["end"],
                    boxes=boxes,
                )
                entities.append(t)
            output = NerOutput(path=text.path, entities=entities, exception="")
            results.append(output)

        return pd.DataFrame(results)
