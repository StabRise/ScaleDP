import tempfile

from pyspark.ml import PipelineModel

from scaledp import (
    ImageDrawBoxes,
    YoloOnnxDetector,
)
from scaledp.enums import Device


def test_yolo_onnx_detector(image_qr_code_df):

    detector = YoloOnnxDetector(
        device=Device.CPU,
        keepInputData=True,
        partitionMap=True,
        numPartitions=0,
        scoreThreshold=0.5,
        task="detect",
        model="StabRise/YOLO_QR_Code_Detection",
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
    result = pipeline.transform(image_qr_code_df)

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
