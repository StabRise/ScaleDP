import tempfile

import pyspark
from pyspark.ml import PipelineModel

from scaledp import ImageDrawBoxes, PdfDataToImage
from scaledp.enums import Device
from scaledp.models.detectors.FaceDetector import FaceDetector
from scaledp.pipeline.PandasPipeline import PandasPipeline, pathSparkFunctions


def test_face_detector(image_face_df):

    detector = FaceDetector(
        device=Device.CPU,
        keepInputData=True,
        partitionMap=True,
        numPartitions=0,
        scoreThreshold=0.25,
        task="detect",
        padding=20,
    )

    draw = ImageDrawBoxes(
        keepInputData=True,
        inputCols=["image", "boxes"],
        filled=False,
        color="green",
        lineWidth=5,
        displayDataList=[],
    )
    # Transform the image dataframe through the OCR stage
    pipeline = PipelineModel(stages=[detector, draw])
    result = pipeline.transform(image_face_df)

    data = result.select("image_with_boxes", "boxes").collect()

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


def test_face_pdf_detector_pandas(face_pdf_file):
    pathSparkFunctions(pyspark)

    pdf = PdfDataToImage(
        inputCol="content",
        outputCol="image",
        pageLimit=1,
    )

    detector = FaceDetector(
        device=Device.CPU,
        keepInputData=True,
        partitionMap=False,
        numPartitions=0,
        scoreThreshold=0.25,
        task="detect",
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
    data = pipeline.fromFile(face_pdf_file)

    # Verify the pipeline result
    assert len(data) == 1, "Expected exactly one result"

    # Save the output image to a temporary file for verification
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
        temp.write(data["image_with_boxes"][0].data)
        temp.close()

        # Print the path to the temporary file
        print("file://" + temp.name)
