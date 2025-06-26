from scaledp.models.recognizers.TesseractOcr import TesseractOcr
from scaledp.pdf.PdfDataToSingleImage import PdfDataToSingleImage


def test_pdf_data_to_image(pdf_df):
    """
    Test the PdfDataToSingleImage transformation functionality.

    This test verifies that:
    1. PDF data is correctly transformed into a single image
    2. The output DataFrame contains the expected number of records
    3. The image field is properly created
    4. The image metadata maintains the correct source path reference

    Args:
        pdf_df: Input DataFrame containing PDF content

    Raises:
        AssertionError: If any of the test conditions fail
    """
    # Initialize the PDF to image transformer
    pdf_data_to_image = PdfDataToSingleImage()
    pdf_data_to_image.setInputCol("content")
    pdf_data_to_image.setOutputCol("image")

    # Perform the transformation
    result = pdf_data_to_image.transform(pdf_df).collect()

    # Test 1: Verify the number of output records
    assert len(result) == 1, "Expected exactly one record in the output DataFrame"

    # Test 2: Verify the presence of the image field
    assert hasattr(result[0], "image"), "Output record missing 'image' field"

    # Test 3: Verify image metadata contains correct source path
    assert (
        result[0].image.path == result[0].path
    ), "Image path does not match source path"

    # Test 4: Verify image field is not None
    assert result[0].image is not None, "Image field should not be None"


def test_pdf_data_to_text(pdf_df):
    """
    Test the conversion of PDF data to text using PdfDataToSingleImage and TesseractOCR.

    This test verifies that:
    1. PDF data is correctly transformed into text via image conversion
    2. The output DataFrame contains the expected number of records
    3. The text field is properly created
    4. The OCR process completes without exceptions
    5. The detected text contains expected content

    Args:
        pdf_df: Input DataFrame containing PDF content

    Raises:
        AssertionError: If any of the test conditions fail
    """
    # Initialize the PDF to image transformer
    pdf_data_to_image = PdfDataToSingleImage()
    pdf_data_to_image.setInputCol("content")
    pdf_data_to_image.setOutputCol("image")

    # Initialize the OCR transformer
    image_to_string = TesseractOcr()

    # Perform the transformations: PDF -> Image -> Text
    result = image_to_string.transform(
        pdf_data_to_image.transform(pdf_df),
    ).collect()

    # Test 1: Verify the number of output records
    assert len(result) == 1, "Expected exactly one record in the output DataFrame"

    # Test 2: Verify the presence of the text field
    assert hasattr(result[0], "text"), "Output record missing 'text' field"

    # Test 3: Verify that OCR completed without exceptions
    assert result[0].text.exception == "", "OCR process encountered an exception"

    # Test 4: Verify the expected content in the OCR result
    expected_text = "UniDoc Medial Center"
    assert (
        expected_text in result[0].text.text
    ), f"Expected text '{expected_text}' not found in OCR result"
