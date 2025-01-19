# Description: Schema Detector Output
from scaledp.utils.dataclass import map_dataclass_to_struct, register_type
from dataclasses import dataclass


@dataclass(order=True)
class ExtractorOutput:
    path: str
    data: str
    type: str
    exception: str = ""
    processing_time: float = 0.0

    @staticmethod
    def get_schema():
        return map_dataclass_to_struct(ExtractorOutput)


register_type(ExtractorOutput, ExtractorOutput.get_schema)
