import tempfile

from pyspark.ml import PipelineModel

from scaledp import (
    ImageDrawBoxes,
    PdfDataToImage,
    TesseractRecognizer,
)
from scaledp.enums import Device, TessLib
from scaledp.models.detectors.CraftTextDetector import CraftTextDetector


def test_craft_detector(image_rotated_df):
    detector = CraftTextDetector(
        device=Device.CPU,
        keepInputData=True,
        partitionMap=True,
        numPartitions=1,
        width=1600,
    )

    ocr = TesseractRecognizer(
        inputCols=["image", "boxes"],
        keepFormatting=False,
        keepInputData=True,
        tessLib=TessLib.PYTESSERACT,
        lang=["eng", "spa"],
        scoreThreshold=0.2,
        scaleFactor=2.0,
        partitionMap=True,
        numPartitions=1,
    )

    draw = ImageDrawBoxes(
        keepInputData=True,
        inputCols=["image", "text"],
        filled=False,
        color="green",
        lineWidth=5,
        displayDataList=["score", "text", "angle"],
    )
    # Transform the image dataframe through the OCR stage
    pipeline = PipelineModel(stages=[detector, ocr, draw])
    result = pipeline.transform(image_rotated_df)

    data = result.collect()

    # Verify the pipeline result
    assert len(data) == 1, "Expected exactly one result"

    # # Check that exceptions is empty
    assert data[0].text.exception == ""

    # Save the output image to a temporary file for verification
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
        temp.write(data[0].image_with_boxes.data)
        temp.close()

        # Print the path to the temporary file
        print("file://" + temp.name)


def test_craft_detector_pdf(pdf_df_extra):
    pdf_data_to_image = PdfDataToImage(inputCol="content", outputCol="image")
    detector = CraftTextDetector(
        device=Device.CPU,
        keepInputData=True,
        partitionMap=True,
        numPartitions=1,
        width=1600,
    )

    ocr = TesseractRecognizer(
        inputCols=["image", "boxes"],
        keepFormatting=False,
        keepInputData=True,
        tessLib=TessLib.PYTESSERACT,
        lang=["eng", "spa"],
        scoreThreshold=0.2,
        partitionMap=True,
        numPartitions=1,
    )

    draw = ImageDrawBoxes(
        keepInputData=True,
        inputCols=["image", "text"],
        filled=False,
        color="green",
        lineWidth=5,
        displayDataList=["score", "text"],
    )
    # Transform the image dataframe through the OCR stage
    pipeline = PipelineModel(stages=[pdf_data_to_image, detector, ocr, draw])
    result = pipeline.transform(pdf_df_extra)

    data = result.collect()

    # Verify the pipeline result
    assert len(data) == 1, "Expected exactly one result"

    # # Check that exceptions is empty
    assert data[0].text.exception == ""

    # Save the output image to a temporary file for verification
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
        temp.write(data[0].image_with_boxes.data)
        temp.close()

        # Print the path to the temporary file
        print("file://" + temp.name)
