import tempfile

from pyspark.ml import PipelineModel
from pyspark.sql import DataFrame

from scaledp import ImageDrawBoxes, TesseractRecognizer, TessLib
from scaledp.models.detectors.DBNetOnnxDetector import DBNetOnnxDetector
from scaledp.models.recognizers.TesseractOcr import TesseractOcr
from scaledp.pdf import PdfAddTextLayer, PdfAssembler, PdfDataToImage, SingleImageToPdf
from scaledp.pipeline.PandasPipeline import PandasPipeline


def test_pdf_assembler(pdf_df: DataFrame) -> None:

    # Initialize pipeline stages
    pdf_data_to_image = PdfDataToImage(
        inputCol="content",
        outputCol="image",
        pageLimit=2,
    )
    ocr = TesseractOcr(
        inputCol="image",
        outputCol="text",
        keepInputData=True,
        tessLib=TessLib.TESSEROCR,
    )

    image_to_pdf = SingleImageToPdf(
        inputCol="image",
        outputCol="pdf",
    )

    pdf_text_layer = PdfAddTextLayer(
        inputCols=["pdf", "text"],
        outputCol="pdf_with_text_layer",
    )

    pdf_assembler = PdfAssembler(
        inputCol="pdf_with_text_layer",
        outputCol="assembled_pdf",
        groupByCol="path",
    )

    # Create and configure the pipeline
    pipeline = PipelineModel(
        stages=[
            pdf_data_to_image,
            ocr,
            image_to_pdf,
            pdf_text_layer,
            pdf_assembler,
        ],
    )

    result = pipeline.transform(pdf_df).collect()

    # Verify the pipeline result
    assert len(result) == 1, "Expected exactly two results"

    assert hasattr(result[0], "assembled_pdf")

    # Verify that there is no exception in the OCR result
    assert (
        result[0].assembled_pdf.exception == ""
    ), "Expected no exception in the OCR result"

    # Create temporary file to store the PDF
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp:
        # Output the file location for user reference
        print(f"PDF saved at: file://{temp.name}")

        # Write PDF data to temporary file
        temp.write(result[0].assembled_pdf.data)


def test_pdf_local_pipeline(patch_spark, pdf_report_file: str) -> None:
    """Test PDF processing using PandasPipeline with local file input."""

    # Initialize pipeline stages
    pdf_data_to_image = PdfDataToImage(
        inputCol="content",
        outputCol="image",
        pageLimit=10,
    )

    text_detector = DBNetOnnxDetector(
        model="StabRise/text_detection_dbnet_ml_v0.1",
        keepInputData=True,
        onlyRotated=False,
    )

    text_recognizer = TesseractRecognizer(
        inputCols=["image", "boxes"],
        outputCol="text",
        keepFormatting=False,
        keepInputData=True,
        tessLib=TessLib.PYTESSERACT,
        lang=["eng", "spa"],
        scoreThreshold=0.2,
        partitionMap=False,
        numPartitions=1,
    )

    draw = ImageDrawBoxes(
        inputCols=["image", "text"],
        outputCol="image_with_boxes",
        lineWidth=2,
        textSize=20,
        displayDataList=[],
        keepInputData=True,
    )

    image_to_pdf = SingleImageToPdf(
        inputCol="image_with_boxes",
        outputCol="pdf",
    )

    pdf_text_layer = PdfAddTextLayer(
        inputCols=["pdf", "text"],
        outputCol="pdf_with_text_layer",
    )

    pdf_assembler = PdfAssembler(
        inputCol="pdf_with_text_layer",
        outputCol="assembled_pdf",
        groupByCol="path",
    )

    # Create and configure the pipeline
    pipeline = PandasPipeline(
        stages=[
            pdf_data_to_image,
            text_detector,
            text_recognizer,
            draw,
            image_to_pdf,
            pdf_text_layer,
            pdf_assembler,
        ],
    )

    # Process the PDF file
    result = pipeline.fromFile(pdf_report_file)

    # Verify pipeline execution and results
    assert result is not None, "Pipeline result should not be None"
    assert "assembled_pdf" in result.columns, "Result should contain 'text' column"
    assert "execution_time" in result.columns, "Result should contain execution timing"

    assert len(result) == 1

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp:
        # Output the file location for user reference
        print(f"PDF saved at: file://{temp.name}")

        # Write PDF data to temporary file
        temp.write(result["assembled_pdf"][0].data)
