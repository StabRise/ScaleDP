from pathlib import Path

from scaledp import DataToImage
from scaledp.pipeline.PandasPipeline import DatasetPd, PandasPipeline, posexplode


def test_local_pipeline(patch_spark, image_file: str) -> None:
    """Test the local pipeline with the DataToImage stage."""
    # Initialize the DataToImage stage
    data_to_image = DataToImage()

    # Initialize the LocalPipeline with the DataToImage stage
    pipeline = PandasPipeline(stages=[data_to_image])

    # Read the image file
    with Path.open(image_file, "rb") as f:
        image_data = f.read()

    # Transform the image data through the pipeline
    result = pipeline.fromBinary(image_data)

    # Verify the pipeline result
    assert result is not None, "Expected a non-None result from the pipeline"
    assert len(result) > 0, "Expected at least one result from the pipeline"


def test_explode(patch_spark, pdf_file: str) -> None:
    """Test the explode function with the DataToImage stage."""
    data = DatasetPd(
        {"content": [("1", "2", "3")], "path": ["memory"], "resolution": [0]},
    )
    print(posexplode(data, "content"))
