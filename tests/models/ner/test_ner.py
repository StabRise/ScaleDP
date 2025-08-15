import pyspark.sql.functions as f
from pyspark.ml import PipelineModel

from scaledp import DataToImage, ImageDrawBoxes
from scaledp.enums import Device
from scaledp.models.ner.Ner import Ner
from scaledp.models.recognizers.TesseractOcr import TesseractOcr
from scaledp.text.TextToDocument import TextToDocument


def test_ner(image_df):
    # Initialize the OCR stage
    ocr = TesseractOcr(keepInputData=True)

    # Initialize the NER stage with the specified model and device
    ner = Ner(
        model="obi/deid_bert_i2b2",
        numPartitions=0,
        inputCols=["text", "text"],
        device=Device.CPU.value,
    )

    # Transform the image dataframe through the OCR and NER stages
    result_df = ner.transform(ocr.transform(image_df))

    # Cache the result for performance
    result = result_df.select("ner", "text").cache()

    # Collect the results
    data = result.collect()

    # Check that exceptions is empty
    assert data[0].text.exception == ""

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
    assert ner_tags.count() > 70


def test_ner_pandas(image_df):
    # Initialize the OCR stage
    ocr = TesseractOcr(keepInputData=True)

    # Initialize the NER stage with the specified model and device
    ner = Ner(
        model="obi/deid_bert_i2b2",
        numPartitions=1,
        inputCols=["text", "text"],
        device=Device.CPU.value,
    )

    # Transform the image dataframe through the OCR and NER stages
    result_df = ner.transform(ocr.transform(image_df))

    # Cache the result for performance
    result = result_df.select("ner", "text").cache()

    # Collect the results
    data = result.collect()

    # Check that exceptions is empty
    assert data[0].text.exception == ""

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
    assert ner_tags.count() > 70


def test_ner_local_pipeline(image_file):
    import pyspark

    from scaledp.pipeline.PandasPipeline import (
        PandasPipeline,
        pathSparkFunctions,
        unpathSparkFunctions,
    )

    # Temporarily replace the UserDefinedFunction
    pathSparkFunctions(pyspark)
    # Initialize the pipeline stages
    data_to_image = DataToImage()
    ocr = TesseractOcr(keepInputData=True)
    ner = Ner(model="obi/deid_bert_i2b2", numPartitions=0, device=Device.CPU.value)
    draw = ImageDrawBoxes(
        keepInputData=True,
        inputCols=["image", "ner"],
        filled=True,
        color="orange",
        displayDataList=["score"],
    )

    # Create the pipeline
    pipeline = PandasPipeline(stages=[data_to_image, ocr, ner, draw])

    # Run the pipeline on the input image file
    result = pipeline.fromFile(image_file)

    # Verify the pipeline result
    assert result is not None
    assert "image_with_boxes" in result.columns
    assert "ner" in result.columns

    # Verify the OCR stage output
    ocr_result = result["text"][0].text
    assert len(ocr_result) > 0

    # Verify the NER stage output
    ner_result = result["ner"][0].entities
    assert len(ner_result) > 0

    # Verify the draw stage output
    draw_result = result["image_with_boxes"][0]
    assert draw_result.exception == ""

    unpathSparkFunctions(pyspark)


def test_ner_with_raw_text(text_df):

    text_to_doc = TextToDocument()
    ner = Ner(model="obi/deid_bert_i2b2")

    pipeline = PipelineModel(stages=[text_to_doc, ner])
    result = pipeline.transform(text_df).cache()

    result.show_ner("ner")

    ner_tags = result.select(f.explode("ner.entities").alias("ner")).select("ner.*")

    # Assert that there are more than 70 NER tags
    assert ner_tags.count() > 50

    result.unpersist()
