from dataclasses import dataclass

from scaledp.utils.dataclass import BinaryT, map_dataclass_to_struct, register_type


@dataclass(order=True)
class PdfDocument:
    path: str
    data: BinaryT
    width: int = None
    height: int = None
    exception: str = ""

    @staticmethod
    def get_schema():
        return map_dataclass_to_struct(PdfDocument)


register_type(PdfDocument, PdfDocument.get_schema)
