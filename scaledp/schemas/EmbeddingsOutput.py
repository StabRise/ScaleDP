# Description: Schema Embeddings Output
from dataclasses import dataclass
from typing import Optional

from scaledp.utils.dataclass import map_dataclass_to_struct, register_type


@dataclass(order=True)
class EmbeddingsOutput:
    path: Optional[str]
    data: Optional[list[float]]
    type: Optional[str]
    exception: Optional[str] = ""
    processing_time: Optional[float] = 0.0

    @staticmethod
    def get_schema():
        return map_dataclass_to_struct(EmbeddingsOutput)


register_type(EmbeddingsOutput, EmbeddingsOutput.get_schema)
