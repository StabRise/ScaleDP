import json
import tempfile

import pyspark.sql.functions as f
import pytest
from pyspark.ml import PipelineModel

from scaledp import ImageDrawBoxes
from scaledp.models.ner.LLMNer import LLMNer
from scaledp.models.recognizers.TesseractOcr import TesseractOcr


def test_llm_ner(image_df):
    pytest.skip("Slow test")
    # Initialize the OCR stage
    ocr = TesseractOcr(keepInputData=True)

    # Initialize the NER stage with the specified model and device
    ner = LLMNer(model="gemini-1.5-flash", numPartitions=0)
    draw = ImageDrawBoxes(
        keepInputData=True,
        inputCols=["image", "ner"],
        filled=False,
        color="grey",
        displayDataList=["entity_group"],
    )

    # Transform the image dataframe through the OCR and NER stages
    pipeline = PipelineModel(stages=[ocr, ner, draw])
    result_df = pipeline.transform(image_df)

    # Cache the result for performance
    result = result_df.select("ner", "text", "image_with_boxes").cache()

    # Collect the results
    data = result.collect()

    # Check that exceptions is empty
    assert data[0].text.exception == ""

    print(
        json.dumps(
            [
                {
                    "text": bbox.text,
                    "score": float(bbox.score),
                    "x": int(bbox.x),
                    "y": int(bbox.y),
                    "width": int(bbox.width),
                    "height": int(bbox.height),
                }
                for bbox in data[0].text.bboxes
            ],
        ),
    )

    # Assert that there is exactly one result
    assert len(data) == 1

    # Assert that the 'ner' field is present in the result
    assert hasattr(data[0], "ner")

    # Display the NER results for debugging
    result.show_ner("ner", 40)

    # Visualize the NER results
    result.visualize_ner()

    # Extract and count the NER tags
    ner_tags = result.select(f.explode("ner.entities").alias("ner")).select("ner.*")

    # Assert that there are more than 70 NER tags
    assert ner_tags.count() > 10

    # Save the output image to a temporary file for verification
    with tempfile.NamedTemporaryFile(suffix=".webp", delete=False) as temp:
        temp.write(data[0].image_with_boxes.data)
        temp.close()

        # Print the path to the temporary file
        print("file://" + temp.name)
