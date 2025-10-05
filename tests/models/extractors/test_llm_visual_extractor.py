import json
from enum import Enum
from typing import Optional

import pytest
from pydantic import BaseModel, Field

from scaledp.image.DataToImage import DataToImage
from scaledp.metrics.object_similarity import calculate_similarity
from scaledp.models.extractors.LLMVisualExtractor import LLMVisualExtractor


class ReceiptItem(BaseModel):
    """Purchased items."""

    name: str
    quantity: float
    price_per_unit: float
    price: float
    product_code: str = Field(description="Product identifier (13 digits)")


class ReceiptItem1(BaseModel):
    """Purchased items."""

    name: str
    quantity: float
    price_per_unit: float
    price: float
    product_code: Optional[str] = Field(
        description="Product identifier (13 digits)",
        default=None,
    )


class Address(BaseModel):
    """Address."""

    country_code: str
    state: str = Field(description="State")
    city: str
    street: str
    house: str


class Address1(BaseModel):
    """Address."""

    country_code: Optional[str] = None
    state: Optional[str] = Field(description="State")
    city: Optional[str] = None
    street: Optional[str] = None
    house: Optional[str] = None


class CompanyType(str, Enum):
    """Receipt type."""

    MARKET = "MARKET"
    PHARMACY = "PHARMACY"
    BANK = "BANK"
    RESTAURANT = "RESTAURANT"
    CAFFE = "CAFFE"
    OTHER = "OTHER"


#  One of: MARKET, PHARMACY, BANK, RESTAURANT, CAFFE, OTHER
class ReceiptSchema(BaseModel):
    """Receipt."""

    company_name: str
    shop_name: str
    company_type: str = Field(
        description="Type of the company.",
        examples=["MARKET", "PHARMACY"],
    )
    address: Address
    tax_id: str
    transaction_date: str = Field(description="Date of the transaction")
    transaction_time: str = Field(description="Time of the transaction")
    total_amount: float
    items: list[ReceiptItem]


class ReceiptSchema1(BaseModel):
    """Receipt."""

    company_name: str
    shop_name: str
    company_type: str = Field(
        description="Type of the company.",
        examples=["MARKET", "PHARMACY"],
    )
    address: Address1
    tax_id: str
    transaction_date: str = Field(description="Date of the transaction")
    transaction_time: str = Field(description="Time of the transaction")
    total_amount: float
    items: list[ReceiptItem1]


def test_llm_visual_extractor_pandas(receipt_file, receipt_json, receipt_json_path):
    pytest.skip("Slow test")
    import pyspark

    from scaledp.pipeline.PandasPipeline import (
        PandasPipeline,
        pathSparkFunctions,
        unpathSparkFunctions,
    )

    # Temporarily replace the UserDefinedFunction
    pathSparkFunctions(pyspark)
    data_to_image = DataToImage()
    extractor = LLMVisualExtractor(
        model="gemini-2.5-flash",
        schema=ReceiptSchema,
        propagateError=True,
    )

    pipeline = PandasPipeline(stages=[data_to_image, extractor])
    # Transform the image dataframe through the OCR and NER stages
    data = pipeline.fromFile(receipt_file)
    data = pipeline.fromFile(receipt_file)

    # Assert that there is exactly one result
    assert len(data) == 1

    # Assert that the 'data' field is present in the result

    print(data["data"][0].data)

    # Check that exceptions is empty
    assert data["data"][0].exception == ""
    receipt = ReceiptSchema.model_validate_json(data["data"][0].data)
    if not receipt_json_path.exists():
        with receipt_json_path.open("w") as f:
            f.write(
                json.dumps(json.loads(data[0].data.data), indent=4, ensure_ascii=False),
            )
    true_receipt = ReceiptSchema.model_validate_json(receipt_json)
    assert calculate_similarity(receipt, true_receipt) > 0.7

    unpathSparkFunctions(pyspark)


def test_llm_visual_extractor(image_receipt_df, receipt_json, receipt_json_path):
    pytest.skip("Slow test")
    extractor = LLMVisualExtractor(
        model="gemini-1.5-flash",
        schema=ReceiptSchema,
        propagateError=False,
        schemaByPrompt=False,
    )

    # Transform the image dataframe through the OCR and NER stages
    result_df = extractor.transform(image_receipt_df)

    # Cache the result for performance
    result = result_df.select("data").cache()

    # Collect the results
    data = result.collect()

    # Assert that there is exactly one result
    assert len(data) == 1

    # Assert that the 'data' field is present in the result
    assert hasattr(data[0], "data")

    # Check that exceptions is empty
    assert data[0].data.exception == ""
    receipt = ReceiptSchema.model_validate_json(data[0].data.json_data)
    if not receipt_json_path.exists():
        with receipt_json_path.open("w") as f:
            f.write(
                json.dumps(
                    json.loads(data[0].data.json_data),
                    indent=4,
                    ensure_ascii=False,
                ),
            )
    true_receipt = ReceiptSchema.model_validate_json(receipt_json)
    similairty = calculate_similarity(receipt, true_receipt)
    print(similairty)
    assert similairty > 0.6

    result.select("data.data.*").show(1, False)


def test_llm_visual_extractor_prompt_schema(
    image_receipt_df,
    receipt_with_null_json,
    receipt_with_null_json_path,
):

    extractor = LLMVisualExtractor(
        model="gemini-2.5-flash",
        schema=ReceiptSchema1,
        propagateError=False,
        schemaByPrompt=True,
    )

    # Transform the image dataframe through the OCR and NER stages
    result_df = extractor.transform(image_receipt_df)

    # Cache the result for performance
    result = result_df.select("data").cache()

    # Collect the results
    data = result.collect()

    # Assert that there is exactly one result
    assert len(data) == 1

    # Assert that the 'data' field is present in the result
    assert hasattr(data[0], "data")

    # Check that exceptions is empty
    assert data[0].data.exception == ""
    print(data[0].data.data)
    receipt = ReceiptSchema1.model_validate_json(data[0].data.json_data)
    if not receipt_with_null_json_path.exists():
        with receipt_with_null_json_path.open("w") as f:
            f.write(
                json.dumps(
                    json.loads(data[0].data.json_data),
                    indent=4,
                    ensure_ascii=False,
                ),
            )
    true_receipt = ReceiptSchema1.model_validate_json(receipt_with_null_json)
    similairty = calculate_similarity(receipt, true_receipt)
    print(similairty)
    assert similairty > 0.6

    result.select("data.data").show(1, False)
