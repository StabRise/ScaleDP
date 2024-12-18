import pytest
import tempfile
from scaledp import DataToImage, ImageDrawBoxes
from scaledp.models.detectors.YoloDetector import YoloDetector
from scaledp.enums import Device


def test_yolo_detector(image_receipt_df):
    detector = YoloDetector(device=Device.CPU, keepInputData=True,
                       model="/Users/nmelnyk/PycharmProjects/daymeai/apiserver/resources/models/detectors/reciept_segmentation_15.12.2024.pt")

    draw = ImageDrawBoxes(keepInputData=True, inputCols=["image", "boxes"],
                          filled=False, color="green", lineWidth=5,
                          displayDataList=['score'])
    # Transform the image dataframe through the OCR stage
    result = draw.transform(detector.transform(image_receipt_df)).cache()

    data = result.collect()

    # Verify the pipeline result
    assert len(data) == 1, "Expected exactly one result"

    # Check that exceptions is empty
    assert data[0].boxes.exception == ""

    # Save the output image to a temporary file for verification
    temp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    temp.write(data[0].image_with_boxes.data)
    temp.close()

    # Print the path to the temporary file
    print("file://" + temp.name)


def test_yolo_detector_local_pipeline(receipt_file):
    from scaledp.pipeline.PandasPipeline import PandasPipeline, pathSparkFunctions, unpathSparkFunctions
    import pyspark

    # Temporarily replace the UserDefinedFunction
    pathSparkFunctions(pyspark)
    # Initialize the pipeline stages
    data_to_image = DataToImage()
    detector = YoloDetector(device=Device.CPU,
                            model="/Users/nmelnyk/PycharmProjects/daymeai/apiserver/resources/models/detectors/reciept_segmentation_15.12.2024.pt")
    draw = ImageDrawBoxes(keepInputData=True, inputCols=["image", "boxes"],
                          filled=False, color="green", lineWidth=5,
                          displayDataList=['score'])

    # Create the pipeline
    pipeline = PandasPipeline(stages=[data_to_image, detector, draw])

    # Run the pipeline on the input image file
    result = pipeline.fromFile(receipt_file)

    # Verify the pipeline result
    assert result is not None
    assert "image_with_boxes" in result.columns
    assert "boxes" in result.columns

    # Verify the Detector stage output
    bboxes = result["boxes"][0].bboxes
    assert len(bboxes) > 0

    # Verify the draw stage output
    draw_result = result["image_with_boxes"][0]
    assert draw_result.exception is ""

    # Save the output image to a temporary file for verification
    temp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    temp.write(draw_result.data)
    temp.close()

    # Print the path to the temporary file
    print("file://" + temp.name)

    unpathSparkFunctions(pyspark)

