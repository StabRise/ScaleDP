import pytest
from sparkpdf.models.recognizers.SuryaOcr import SuryaOcr
from sparkpdf.enums import Device


def test_surya_ocr_class(image_line):
    # Test SuryatOcr
    result1 = SuryaOcr(keepFormatting=False, partitionMap=False, device=Device.CUDA).transform_udf(image_line)
    assert result1.exception == "", "Expected no exception for keepFormatting=False"
    assert "24/11/16 08:12:24 WARN Utils: Service 'SparkUI' could not bind on port 4043. Attempting port 4044." in result1.text


def test_surya_ocr_pandas_udf(image_line_df):
    ocr = SuryaOcr(partitionMap=True, device=Device.CUDA, keepFormatting=True)

    # Transform the image dataframe through the OCR stage
    result = ocr.transform(image_line_df).collect()

    # Verify the pipeline result
    assert len(result) == 1, "Expected exactly one result"

    # Verify the detected text contains the expected substring
    assert "24/11/16 08:12:24 WARN Utils: Service 'SparkUI' could not bind on port 4043. Attempting port 4044." in result[0].text.text
