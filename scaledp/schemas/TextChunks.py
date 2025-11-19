from dataclasses import dataclass
from typing import Optional

from scaledp.utils.dataclass import map_dataclass_to_struct, register_type


@dataclass(order=True)
class TextChunks:
    path: Optional[str]
    chunks: Optional[list[str]]
    exception: Optional[str] = ""
    processing_time: Optional[float] = 0.0

    @staticmethod
    def get_schema():
        return map_dataclass_to_struct(TextChunks)


register_type(TextChunks, TextChunks.get_schema)
