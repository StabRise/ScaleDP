import logging
import tempfile

from pyspark.ml import PipelineModel

from scaledp import (
    ImageDrawBoxes,
    TesseractRecognizer,
)
from scaledp.enums import TessLib
from scaledp.models.detectors.DBNetOnnxDetector import DBNetOnnxDetector
from scaledp.pdf.PdfDataToImage import PdfDataToImage
from scaledp.pipeline.PandasPipeline import PandasPipeline


def test_dbnet_detector(image_rotated_text_df):

    detector = DBNetOnnxDetector(
        model="StabRise/text_detection_dbnet_ml_v0.2",
        keepInputData=True,
        onlyRotated=False,
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
    result = pipeline.transform(image_rotated_text_df)

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
        logging.info("file://" + temp.name)


def test_dbnet_detector_rotated_text_pdf(
    patch_spark,
    extraction_conditions_pdf_file,
    extraction_conditions_rotated_probes,
):
    """Rotated, skewed and stacked text is detected on a rendered PDF page,
    both for text-layer and rasterised (image-only) cells.
    """
    pipeline = PandasPipeline(
        stages=[
            PdfDataToImage(inputCol="content", outputCol="image"),
            DBNetOnnxDetector(
                model="StabRise/text_detection_dbnet_ml_v0.2",
                keepInputData=True,
                onlyRotated=False,
            ),
        ],
    )
    result = pipeline.fromFile(extraction_conditions_pdf_file)

    assert len(result) == 1, "Expected exactly one page"
    boxes = result["boxes"][0]
    assert boxes.exception == ""

    # Probe bboxes are in points, detections are in rendered image pixels
    scale = result["image"][0].width / 595.28

    for probe in extraction_conditions_rotated_probes:
        x0, y0, x1, y1 = (v * scale for v in probe["bbox"])
        # Rotated boxes keep unrotated width/height, so match on box center
        found = [
            box
            for box in boxes.bboxes
            if x0 <= box.x + box.width / 2 <= x1 and y0 <= box.y + box.height / 2 <= y1
        ]
        assert found, f"{probe['id']} ({probe['orientation']}) not detected"

        angles = [box.angle for box in found]
        if probe["orientation"] in ("rot90", "rot270"):
            assert any(
                abs(abs(a) % 180 - 90) < 10 for a in angles
            ), f"{probe['id']}: expected vertical box, got angles {angles}"
        elif probe["orientation"] == "skew":
            # Cells are rotated by 15 degrees counter-clockwise
            assert any(
                -25 < a < -5 for a in angles
            ), f"{probe['id']}: expected skewed box, got angles {angles}"
