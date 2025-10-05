import base64
import json
import logging
from types import MappingProxyType
from typing import Any

from pyspark import keyword_only
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from scaledp.params import HasLLM, HasPrompt, HasSchema

from .BaseVisualExtractor import BaseVisualExtractor


class LLMVisualExtractor(BaseVisualExtractor, HasLLM, HasSchema, HasPrompt):

    defaultParams = MappingProxyType(
        {
            "inputCol": "image",
            "outputCol": "data",
            "keepInputData": True,
            "model": "gemini-2.5-flash",
            "apiBase": None,
            "apiKey": None,
            "numPartitions": 1,
            "pageCol": "page",
            "pathCol": "path",
            "prompt": """Please extract data from the scanned image as json.
         Date format is yyyy-mm-dd""",
            "systemPrompt": "You are data extractor from the scanned images.",
            "delay": 30,
            "maxRetry": 6,
            "propagateError": False,
            "temperature": 1.0,
            "schemaByPrompt": True,
        },
    )

    @keyword_only
    def __init__(self, **kwargs: Any) -> None:
        super(LLMVisualExtractor, self).__init__()
        self._setDefault(**self.defaultParams)
        self._set(**kwargs)
        self.pipeline = None

    def call_extractor(self, images, params):
        from openai import RateLimitError

        client = self.getOIClient()
        results = []

        @retry(
            retry=retry_if_exception_type(RateLimitError),
            wait=wait_random_exponential(min=1, max=self.getDelay()),
            stop=stop_after_attempt(self.getMaxRetry()),
        )
        def completion_with_backoff(**kwargs: Any):
            logging.info("Calling LLM API")
            return client.beta.chat.completions.parse(**kwargs)

        kwargs = {}

        content = [{"type": "text", "text": params["prompt"]}]
        if self.getSchemaByPrompt():
            content.append(
                {
                    "type": "text",
                    "text": "Schema for the output json: "
                    + self.getSchema()
                    + " Always return valid json. Do not include schema to the output.",
                },
            )
        else:
            kwargs["response_format"] = self.getPaydanticSchema()

        for image in images:
            image_decoded = base64.b64encode(image.data).decode("utf-8")
            completion = completion_with_backoff(
                model=params["model"],
                messages=[
                    {
                        "role": "system",
                        "content": params["systemPrompt"],
                    },
                    {
                        "role": "user",
                        "content": [
                            *content,
                            *[
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_decoded}",
                                    },
                                },
                            ],
                        ],
                    },
                ],
                temperature=self.getTemperature(),
                **kwargs,
            )

            result = (
                completion.choices[0]
                .message.content.replace("```json", "")
                .replace("```", "")
            )

            data = json.loads(result)
            results.append(
                self.getOutputClass()(
                    path=image.path,
                    json_data=json.dumps(data, indent=4, ensure_ascii=False),
                    type="LLMVisualExtractor",
                    data=data,
                    exception="",
                ),
            )
        return results
