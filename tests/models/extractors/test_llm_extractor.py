from pydantic import BaseModel, Field

from scaledp.enums import TessLib
from scaledp.models.extractors.LLMExtractor import LLMExtractor
from scaledp.models.recognizers.TesseractOcr import TesseractOcr


class ReceiptItem(BaseModel):
    """Purchased items."""

    name: str
    quantity: float
    price_per_unit: float
    price: float
    product_code: str


class ReceiptSchema(BaseModel):
    """Receipt."""

    company_name: str
    shop_name: str
    address: str
    tax_id: str
    transaction_date: str = Field(description="Date of the transaction")
    total_amount: float
    items: list[ReceiptItem]


def test_llm_extractor(image_receipt_df):
    # Initialize the OCR stage
    ocr = TesseractOcr(
        keepInputData=True,
        lang=["ukr", "eng"],
        keepFormatting=True,
        tessLib=TessLib.PYTESSERACT,
    )

    # Initialize the NER stage with the specified model and device
    extractor = LLMExtractor(model="gemini-2.5-flash", schema=ReceiptSchema)

    # Transform the image dataframe through the OCR and NER stages
    result_df = extractor.transform(ocr.transform(image_receipt_df))

    # Cache the result for performance
    result = result_df.select("data", "text").cache()

    # Collect the results
    data = result.collect()

    # Check that exceptions is empty
    assert data[0].text.exception == ""

    # Assert that there is exactly one result
    assert len(data) == 1

    # Assert that the 'data' field is present in the result
    assert hasattr(data[0], "data")

    # Check that exceptions is empty
    assert data[0].data.exception == ""

    result.select("data.data").show(1, False)
