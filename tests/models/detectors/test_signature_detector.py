import tempfile

import pyspark
from scaledp.pipeline.PandasPipeline import PandasPipeline, pathSparkFunctions
from pyspark.ml import PipelineModel

from scaledp import (
    ImageDrawBoxes,
    PdfDataToImage,
    SignatureDetector,
)
from scaledp.enums import Device


def test_signature_detector(image_signature_df):

    detector = SignatureDetector(
        device=Device.CPU,
        keepInputData=True,
        partitionMap=True,
        numPartitions=0,
        task="detect",
        model="StabRise/signature_detection",
    )

    draw = ImageDrawBoxes(
        keepInputData=True,
        inputCols=["image", "boxes"],
        filled=False,
        color="green",
        lineWidth=5,
        displayDataList=["score", "angle"],
    )
    # Transform the image dataframe through the OCR stage
    pipeline = PipelineModel(stages=[detector, draw])
    result = pipeline.transform(image_signature_df)

    data = result.collect()

    # Verify the pipeline result
    assert len(data) == 1, "Expected exactly one result"

    # # Check that exceptions is empty
    assert data[0].boxes.exception == ""

    # Save the output image to a temporary file for verification
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
        temp.write(data[0].image_with_boxes.data)
        temp.close()

        # Print the path to the temporary file
        print("file://" + temp.name)


def test_signature_pdf_detector(signatures_pdf_df):

    pipeline = PipelineModel(
        stages=[
            PdfDataToImage(outputCol="image"),
            SignatureDetector(
                device=Device.CPU,
                keepInputData=True,
                outputCol="signatures",
                scoreThreshold=0.20,
                model="StabRise/signature_detection",
            ),
            ImageDrawBoxes(
                keepInputData=True,
                inputCols=["image", "signatures"],
                filled=True,
                color="black",
            ),
        ],
    )
    result = pipeline.transform(signatures_pdf_df)

    data = result.collect()

    # Verify the pipeline result
    assert len(data) == 1, "Expected exactly one result"

    # # Check that exceptions is empty
    assert data[0].boxes.exception == ""

    # Save the output image to a temporary file for verification
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
        temp.write(data[0].image_with_boxes.data)
        temp.close()

        # Print the path to the temporary file
        print("file://" + temp.name)


def test_signature_pdf_detector_pandas(signatures_pdf_file):
    pathSparkFunctions(pyspark)

    # pdf = PdfDataToSingleImage(inputCol="content", outputCol="image",
    #                            keepInputData=True)

    pdf = PdfDataToImage(
        inputCol="content",
        outputCol="image",
        pageLimit=1,
    )

    detector = SignatureDetector(
        device=Device.CPU,
        keepInputData=True,
        partitionMap=False,
        numPartitions=0,
        scoreThreshold=0.25,
        task="detect",
        model="StabRise/signature_detection",
    )

    draw = ImageDrawBoxes(
        keepInputData=True,
        inputCols=["image", "boxes"],
        filled=False,
        color="green",
        lineWidth=5,
        displayDataList=["score", "angle"],
    )
    # Transform the image dataframe through the OCR stage
    pipeline = PandasPipeline(stages=[pdf, detector, draw])
    data = pipeline.fromFile(signatures_pdf_file)

    # Verify the pipeline result
    assert len(data) == 1, "Expected exactly one result"

    # Save the output image to a temporary file for verification
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
        temp.write(data["image_with_boxes"][0].data)
        temp.close()

        # Print the path to the temporary file
        print("file://" + temp.name)
