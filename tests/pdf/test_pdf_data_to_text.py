from pathlib import Path

from pyspark.sql import DataFrame

from scaledp.pdf.PdfDataToText import PdfDataToText


def test_pdf_data_to_text(pdf_df: DataFrame) -> None:
    # Initialize the PdfDataToText stage with specific input and output columns
    pdf_data_to_text = PdfDataToText(inputCol="content", outputCol="document")

    # Transform the PDF dataframe to text dataframe
    result = pdf_data_to_text.transform(pdf_df).collect()

    # Verify the pipeline result
    assert len(result) == 2, "Expected exactly two results"

    # Verify the presence of the 'document' field in the result
    assert hasattr(result[0], "document"), "Expected 'document' field in the result"

    # Verify that there is no exception in the result
    assert result[0].document.exception == "", "Expected no exception in the result"

    # Verify that the text was extracted
    assert result[0].document.text != "", "Expected non-empty text in the result"

    # Verify that the detected text contains the expected substring
    assert (
        "UniDoc Medial Center" in result[0].document.text
    ), "Expected 'UniDoc Medical Center' in the detected text"

    # Verify that bounding boxes were extracted
    assert len(result[0].document.bboxes) > 0, "Expected bounding boxes in the result"

    # Verify that the document type is set correctly
    assert result[0].document.type == "pdf", "Expected document type to be 'pdf'"


def test_pdf_data_to_text_class(pdf_file: str) -> None:
    """Test the PdfDataToText class with the UDF transform method."""
    # Read the PDF file
    with Path.open(pdf_file, "rb") as f:
        data = f.read()

    pdf_to_text = PdfDataToText()

    # Transform the PDF data to text and verify the result
    result = list(pdf_to_text.transform_udf(data, "path"))
    assert len(result) == 2, "Expected 2 pages from the PDF file"

    # Verify the content of the first page
    first_page = result[0]
    assert isinstance(first_page.text, str), "Expected text to be a string"
    assert len(first_page.bboxes) > 0, "Expected bounding boxes for words"
    assert first_page.type == "pdf", "Expected document type to be 'pdf'"
    assert first_page.exception == "", "Expected no exception"

    # Test with None input and verify the result
    result_none = list(pdf_to_text.transform_udf(None, "path"))
    assert len(result_none) == 1, "Expected 1 error document from None input"
    assert result_none[0].exception != "", "Expected error message for None input"
    assert result_none[0].text == "", "Expected empty text for None input"
    assert len(result_none[0].bboxes) == 0, "Expected no bounding boxes for None input"


def test_pdf_data_to_text_invalid_pdf(tmp_path: Path) -> None:
    """Test the PdfDataToText with invalid PDF data."""
    # Create an invalid PDF file
    invalid_pdf = tmp_path / "invalid.pdf"
    invalid_pdf.write_bytes(b"This is not a PDF file")

    with invalid_pdf.open("rb") as f:
        data = f.read()

    pdf_to_text = PdfDataToText()

    # Transform the invalid PDF data and verify the result
    result = list(pdf_to_text.transform_udf(data, "path"))
    assert len(result) == 1, "Expected 1 error document from invalid PDF"
    assert result[0].exception != "", "Expected error message for invalid PDF"
    assert result[0].text == "", "Expected empty text for invalid PDF"
    assert len(result[0].bboxes) == 0, "Expected no bounding boxes for invalid PDF"
