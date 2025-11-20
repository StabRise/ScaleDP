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


def test_text_embeddings_with_text_chunks(df_text_chunks):
    """Test TextEmbeddings with TextChunks schema input."""
    # Initialize the TextEmbeddings stage with TextChunks input
    text_embeddings = TextEmbeddings(
        model="all-MiniLM-L6-v2",
        inputCol="text_chunks_col",
        outputCol="embeddings",
        device=Device.CPU.value,
        batchSize=2,
    )

    # Create a pipeline with the TextEmbeddings stage
    pipeline = PipelineModel(stages=[text_embeddings])

    result_df = pipeline.transform(df_text_chunks)

    # Cache the result for performance
    result = result_df.select("embeddings", "text_chunks_col").cache()

    # Collect the results
    data = result.collect()

    # Should have 5 rows (2 + 2 + 1 from exploded chunks)
    assert len(data) == 5

    # Check that exceptions are empty
    assert all(row.embeddings.exception == "" for row in data)

    # Verify the embeddings are not empty and have correct type
    for row in data:
        assert row.embeddings.data is not None
        assert len(row.embeddings.data) > 0
        assert row.embeddings.type == "text_chunk"
        # Verify path is preserved from TextChunks
        assert row.embeddings.path in ["file1.txt", "file2.txt", "file3.txt"]


def test_text_embeddings_with_text_chunks_pandas(df_text_chunks):
    """Test TextEmbeddings with TextChunks schema input using pandas_udf."""
    # Initialize the TextEmbeddings stage with partitionMap=True (pandas_udf)
    text_embeddings = TextEmbeddings(
        model="all-MiniLM-L6-v2",
        inputCol="text_chunks_col",
        outputCol="embeddings",
        device=Device.CPU.value,
        partitionMap=True,
        batchSize=2,
    )

    # Create a pipeline with the TextEmbeddings stage
    pipeline = PipelineModel(stages=[text_embeddings])

    result_df = pipeline.transform(df_text_chunks)

    # Cache the result for performance
    result = result_df.select("embeddings", "text_chunks_col").cache()

    # Collect the results
    data = result.collect()

    # Should have 5 rows (2 + 2 + 1 from pandas_udf processing)
    assert len(data) == 5

    # Check that exceptions are empty
    assert all(row.embeddings.exception == "" for row in data)

    # Verify the embeddings are not empty and have correct type
    for row in data:
        assert row.embeddings.data is not None
        assert len(row.embeddings.data) > 0
        assert row.embeddings.type == "text_chunk"
        # Verify path is preserved from TextChunks
        assert row.embeddings.path in ["file1.txt", "file2.txt", "file3.txt"]
