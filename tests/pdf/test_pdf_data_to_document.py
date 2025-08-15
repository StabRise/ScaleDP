import tempfile

from pyspark.ml import PipelineModel

from scaledp import ImageDrawBoxes
from scaledp.models.recognizers.TesseractOcr import TesseractOcr
from scaledp.pdf.PdfDataToDocument import PdfDataToDocument
from scaledp.pdf.PdfDataToSingleImage import PdfDataToSingleImage


def test_pdf_data_to_document(pdf_df):
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
    pdf_data_to_image = PdfDataToSingleImage(
        inputCol="content",
        outputCol="image",
        keepInputData=True,
    )
    pdf_data_to_document = PdfDataToDocument(inputCol="content", outputCol="text")
    draw = ImageDrawBoxes(inputCols=["image", "text"], filled=False, color="red")

    # Initialize the OCR transformer
    ocr = TesseractOcr(
        inputCol="image",
        outputCol="text",
        bypassCol="text",
        keepInputData=True,
    )

    pipeline = PipelineModel(
        stages=[pdf_data_to_image, pdf_data_to_document, ocr, draw],
    )
    result_df = pipeline.transform(pdf_df)

    # Perform the transformations: PDF -> Image -> Text
    result = result_df.collect()

    # Test 1: Verify the number of output records
    assert len(result) == 1, "Expected exactly one record in the output DataFrame"

    # Test 2: Verify the presence of the text field
    assert hasattr(result[0], "text"), "Output record missing 'text' field"

    # Test 3: Verify that OCR completed without exceptions
    assert result[0].text.exception == "", "OCR process encountered an exception"

    assert hasattr(result[0], "image_with_boxes")

    # Save the output image to a temporary file for verification
    with tempfile.NamedTemporaryFile(suffix=".webp", delete=False) as temp:
        temp.write(result[0].image_with_boxes.data)
        temp.close()

        # Print the path to the temporary file
        print("file://" + temp.name)


def test_image_pdf_data_to_document(image_pdf_df):
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
    pdf_data_to_image = PdfDataToSingleImage(
        inputCol="content",
        outputCol="image",
        keepInputData=True,
    )
    pdf_data_to_document = PdfDataToDocument(inputCol="content", outputCol="text")
    draw = ImageDrawBoxes(inputCols=["image", "text"], filled=False, color="red")

    # Initialize the OCR transformer
    ocr = TesseractOcr(
        inputCol="image",
        outputCol="text",
        bypassCol="text",
        keepInputData=True,
    )

    pipeline = PipelineModel(
        stages=[pdf_data_to_image, pdf_data_to_document, ocr, draw],
    )
    result_df = pipeline.transform(image_pdf_df)

    # Perform the transformations: PDF -> Image -> Text
    result = result_df.collect()

    # Test 1: Verify the number of output records
    assert len(result) == 1, "Expected exactly one record in the output DataFrame"

    # Test 2: Verify the presence of the text field
    assert hasattr(result[0], "text"), "Output record missing 'text' field"

    # Test 3: Verify that OCR completed without exceptions
    assert result[0].text.exception == "", "OCR process encountered an exception"

    # Test 4: Verify the expected content in the OCR result
    expected_text = "YOUR COMPANY   123 YOUR STREET"
    assert (
        expected_text in result[0].text.text
    ), f"Expected text '{expected_text}' not found in OCR result"

    assert hasattr(result[0], "image_with_boxes")

    # Save the output image to a temporary file for verification
    with tempfile.NamedTemporaryFile(suffix=".webp", delete=False) as temp:
        temp.write(result[0].image_with_boxes.data)
        temp.close()

        # Print the path to the temporary file
        print("file://" + temp.name)


def test_pdf_local_pipeline(pdf_file):
    import pyspark

    from scaledp.pipeline.PandasPipeline import (
        PandasPipeline,
        pathSparkFunctions,
        unpathSparkFunctions,
    )

    # Temporarily replace the UserDefinedFunction
    pathSparkFunctions(pyspark)
    # Initialize the pipeline stages

    pdf_data_to_image = PdfDataToSingleImage(
        inputCol="content",
        outputCol="image",
        keepInputData=True,
    )
    pdf_data_to_document = PdfDataToDocument(inputCol="content", outputCol="text")

    # Initialize the OCR transformer
    ocr = TesseractOcr(
        inputCol="image",
        outputCol="text",
        bypassCol="text",
        keepInputData=True,
    )

    # Create the pipeline
    pipeline = PandasPipeline(
        stages=[
            pdf_data_to_image,
            pdf_data_to_document,
            ocr,
        ],
    )

    # Run the pipeline on the input image file
    result = pipeline.fromFile(pdf_file)

    # Verify the pipeline result
    assert result is not None
    assert "text" in result.columns

    # Verify the OCR stage output
    ocr_result = result["text"][0].text
    assert len(ocr_result) > 0

    print(result["execution_time"])

    assert "execution_time" in result.columns

    unpathSparkFunctions(pyspark)
