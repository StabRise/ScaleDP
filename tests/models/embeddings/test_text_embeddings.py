from pyspark.ml import PipelineModel

from scaledp.enums import Device
from scaledp.models.embeddings.TextEmbeddings import TextEmbeddings


def test_text_embeddings_pipeline(text_df):

    # Initialize the TextEmbeddings stage
    text_embeddings = TextEmbeddings(
        model="all-MiniLM-L6-v2",
        inputCol="value",
        outputCol="embeddings",
        device=Device.CPU.value,
        batchSize=2,
    )

    # Create a pipeline with the TextEmbeddings stage
    pipeline = PipelineModel(stages=[text_embeddings])

    result_df = pipeline.transform(text_df)

    # Cache the result for performance
    result = result_df.select("embeddings", "value").cache()

    # Collect the results
    data = result.collect()

    # Check that exceptions are empty
    assert all(row.embeddings.exception == "" for row in data)

    # Assert that there is at least one result
    assert len(data) > 0

    # Assert that the 'embeddings' field is present in the result
    assert hasattr(data[0], "embeddings")

    # Verify the embeddings are not empty
    for row in data:
        assert row.embeddings.data is not None
        assert len(row.embeddings.data) > 0


def test_text_embeddings_pipeline_pandas(text_df):

    # Initialize the TextEmbeddings stage
    text_embeddings = TextEmbeddings(
        model="all-MiniLM-L6-v2",
        inputCol="value",
        outputCol="embeddings",
        device=Device.CPU.value,
        partitionMap=True,
        batchSize=2,
    )

    # Create a pipeline with the TextEmbeddings stage
    pipeline = PipelineModel(stages=[text_embeddings])

    result_df = pipeline.transform(text_df)

    # Cache the result for performance
    result = result_df.select("embeddings", "value").cache()

    # Collect the results
    data = result.collect()

    # Check that exceptions are empty
    assert all(row.embeddings.exception == "" for row in data)

    # Assert that there is at least one result
    assert len(data) > 0

    # Assert that the 'embeddings' field is present in the result
    assert hasattr(data[0], "embeddings")

    # Verify the embeddings are not empty
    for row in data:
        assert row.embeddings.data is not None
        assert len(row.embeddings.data) > 0
