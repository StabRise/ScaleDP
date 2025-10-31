import tempfile

from pyspark.ml.pipeline import PipelineModel

from scaledp.enums import PSM
from scaledp.image.ImageCropBoxes import ImageCropBoxes
from scaledp.models.recognizers.TesseractOcr import TesseractOcr


def test_image_crop_boxes_ocr(image_df):

    # Initialize the OCR stage with specific parameters
    ocr = TesseractOcr(
        keepInputData=True,
        scoreThreshold=0.5,
        psm=PSM.SPARSE_TEXT.value,
        scaleFactor=2.0,
    )

    # Initialize the ImageCropBoxes stage
    crop = ImageCropBoxes(
        inputCols=["image", "text"],
        limit=2,
    )

    # Create the pipeline with the OCR and ImageDrawBoxes stages
    pipeline = PipelineModel(stages=[ocr, crop])

    # Run the pipeline on the input image dataframe
    result = pipeline.transform(image_df).collect()

    # Verify the pipeline result
    assert len(result) == 2
    assert hasattr(result[0], "cropped_image")

    # Save the output image to a temporary file for verification
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
        temp.write(result[0].cropped_image.data)
        temp.close()

        # Print the path to the temporary file
        print("file://" + temp.name)

    # Verify the OCR stage output
    ocr_result = result[0].text
    assert len(ocr_result) > 0

    # Verify the draw stage output
    cropped_image = result[0].cropped_image
    assert cropped_image.exception == ""
