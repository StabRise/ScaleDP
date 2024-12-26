# Description: Schema Detector Output
from scaledp.utils.dataclass import map_dataclass_to_struct, register_type
from dataclasses import dataclass
from scaledp.schemas.Box import Box

@dataclass(order=True)
class ExtractorOutput:
    path: str
    data: str
    type: str
    exception: str = ""

    @staticmethod
    def get_schema():
        return map_dataclass_to_struct(ExtractorOutput)

register_type(ExtractorOutput, ExtractorOutput.get_schema)