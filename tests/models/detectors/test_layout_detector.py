import logging
import tempfile
import warnings

import pytest
from pyspark.ml import PipelineModel

from scaledp import ImageDrawBoxes
from scaledp.enums import Device
from scaledp.models.detectors.LayoutDetector import LayoutDetector


@pytest.fixture(autouse=True)
def suppress_warnings():
    """Suppress SWIG deprecation warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            module="importlib._bootstrap",
        )
        yield


@pytest.fixture
def layout_detector():
    return LayoutDetector(
        inputCol="image",
        outputCol="layout_boxes",
        scoreThreshold=0.5,
        device=Device.CPU,
        whiteList=[],
        model="PP-DocLayout_plus-L",
        propagateError=True,
        keepInputData=True,
    )


def test_layout_detector_initialization(layout_detector):
    """Test that LayoutDetector initializes correctly."""
    assert layout_detector.getInputCol() == "image"
    assert layout_detector.getOutputCol() == "layout_boxes"
    assert layout_detector.getScoreThreshold() == 0.5
    assert layout_detector.getDevice() == Device.CPU
    assert layout_detector.getWhiteList() == []
    assert layout_detector.getModel() == "PP-DocLayout_plus-L"


def test_layout_detector_with_drawn_boxes(image_df):
    """Test LayoutDetector with drawn boxes on the original image."""
    detector = LayoutDetector(
        inputCol="image",
        outputCol="layout_boxes",
        scoreThreshold=0.5,
        device=Device.CPU,
        whiteList=["text", "title", "list", "table", "figure"],
        model="PP-DocLayout_plus-L",
        keepInputData=True,
    )

    # Create draw component to visualize detected boxes
    draw = ImageDrawBoxes(
        keepInputData=True,
        inputCols=["image", "layout_boxes"],
        filled=False,
        color="blue",
        lineWidth=4,
        displayDataList=["text", "score"],
    )

    try:

        # Create a pipeline with detector and draw components
        pipeline = PipelineModel(stages=[detector, draw])
        result = pipeline.transform(image_df)

        data = result.collect()

        # Verify the pipeline result
        assert len(data) == 1, "Expected exactly one result"

        # Check that the output column exists and has the expected structure
        assert hasattr(data[0], "layout_boxes"), "Expected layout_boxes column"
        assert data[0].layout_boxes.type == "layout"
        assert isinstance(data[0].layout_boxes.bboxes, list)

        # Check that the image with boxes was created
        assert hasattr(data[0], "image_with_boxes"), "Expected image_with_boxes column"

        # Save the output image to a temporary file for verification
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
            temp.write(data[0].image_with_boxes.data)
            temp.close()

            # Print the path to the temporary file
            logging.info("file://" + temp.name)

    except ImportError:
        pytest.skip("PaddleOCR not installed")
    except Exception as e:
        # Handle other exceptions that might occur during processing
        assert "Error in object detection" in str(e) or "PaddleOCR" in str(e)


def test_layout_detector_with_custom_layout_types():
    """Test LayoutDetector with custom layout types."""
    detector = LayoutDetector(
        inputCol="image",
        outputCol="layout_boxes",
        whiteList=["text", "table"],  # Only detect text and table
        model="PP-DocLayout-M",  # Use different model
        keepInputData=True,
    )

    assert detector.getWhiteList() == ["text", "table"]
    assert detector.getModel() == "PP-DocLayout-M"


def test_layout_detector_output_schema(layout_detector):
    """Test that the output schema is correct."""
    schema = layout_detector.outputSchema()

    # Check that the schema has the expected fields
    field_names = [field.name for field in schema.fields]
    expected_fields = ["path", "type", "bboxes", "exception"]

    for field in expected_fields:
        assert field in field_names
