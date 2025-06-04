import tempfile

from scaledp.pdf.SingleImageToPdf import SingleImageToPdf


def test_image_to_pdf(image_df):
    """
    Test function to convert image DataFrame to PDF format.

    Args:
        image_df: DataFrame containing image data to be converted

    Side Effects:
        - Creates a temporary PDF file with the converted image
        - Prints the local file path of the created PDF
    """
    # Initialize the PDF converter
    image_to_pdf = SingleImageToPdf()

    # Transform the image DataFrame to PDF format and cache for performance
    result_df = image_to_pdf.transform(image_df).cache()

    # Extract PDF data from the transformed DataFrame
    result = result_df.select("pdf").collect()

    # Check if exactly one record was processed
    assert len(result) == 1, "Expected single result record"

    # Verify PDF field exists in result
    assert hasattr(result[0], "pdf"), "PDF field missing in result"

    # Check for any conversion errors
    assert (
        result[0].pdf.exception == ""
    ), f"PDF conversion error: {result[0].pdf.exception}"

    # Create temporary file to store the PDF
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp:
        # Output the file location for user reference
        print(f"PDF saved at: file://{temp.name}")

        # Write PDF data to temporary file
        temp.write(result[0].pdf.data)
