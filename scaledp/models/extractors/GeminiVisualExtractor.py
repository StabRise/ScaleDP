import json
from types import MappingProxyType
from typing import Any

from pyspark import keyword_only

from scaledp.params import HasLLM, HasPrompt, HasSchema
from scaledp.schemas.ExtractorOutput import ExtractorOutput
from scaledp.utils.pydantic_shema_utils import json_schema_to_model

from .BaseVisualExtractor import BaseVisualExtractor


class GeminiVisualExtractor(BaseVisualExtractor, HasLLM, HasSchema, HasPrompt):

    defaultParams = MappingProxyType(
        {
            "inputCol": "image",
            "outputCol": "data",
            "keepInputData": True,
            "model": "gemini-2.5-flash",
            "apiBase": "",
            "apiKey": "",
            "numPartitions": 1,
            "pageCol": "page",
            "pathCol": "path",
            "prompt": """Please extract data from the scanned image as json.""",
        },
    )

    @keyword_only
    def __init__(self, **kwargs: Any) -> None:
        super(GeminiVisualExtractor, self).__init__()
        self._setDefault(**self.defaultParams)
        self._set(**kwargs)
        self.pipeline = None

    def call_extractor(self, images, params):
        import base64

        import google.generativeai as genai

        schema = json.loads(params["schema"])
        schema = json_schema_to_model(schema, schema.get("$defs", {}))
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "application/json",
            "response_schema": schema,
        }

        genai.configure(api_key=params["apiKey"])

        model = genai.GenerativeModel(
            model_name=params["model"],
            generation_config=generation_config,
        )

        schema = json.loads(params["schema"])
        schema = json_schema_to_model(schema, schema.get("$defs", {}))

        results = []

        for image in images:
            image_decoded = base64.b64encode(image.data).decode("utf-8")
            prompt = params["prompt"].format(schema=schema)
            response = model.generate_content(
                [{"mime_type": "image/png", "data": image_decoded}, prompt],
                stream=False,
            )
            results.append(
                ExtractorOutput(
                    path=image.path,
                    data=response.text,
                    type="GeminiVisualExtractor",
                    exception="",
                ),
            )
        return results
