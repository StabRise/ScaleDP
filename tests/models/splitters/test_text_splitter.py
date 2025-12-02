from pyspark.ml import PipelineModel

from scaledp.models.splitters.TextSplitter import TextSplitter
from scaledp.schemas.Document import Document


def test_text_splitter_initialization():
    """Test that TextSplitter can be initialized with default parameters."""
    text_splitter = TextSplitter()

    assert text_splitter.getOrDefault("chunk_size") == 500
    assert text_splitter.getOrDefault("inputCol") == "document"
    assert text_splitter.getOrDefault("outputCol") == "chunks"


def test_text_splitter_custom_parameters():
    """Test that TextSplitter can be initialized with custom parameters."""
    text_splitter = TextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        inputCol="my_document",
        outputCol="text_chunks",
    )

    assert text_splitter.getOrDefault("chunk_size") == 1000
    assert text_splitter.getOrDefault("chunk_overlap") == 100
    assert text_splitter.getOrDefault("inputCol") == "my_document"
    assert text_splitter.getOrDefault("outputCol") == "text_chunks"


def test_text_splitter_split_method():
    """Test the split method directly on a Document."""
    text_splitter = TextSplitter(chunk_size=100, chunk_overlap=10)

    # Create a test document
    long_text = " ".join(["This is a test sentence."] * 20)
    document = Document(path="test.txt", text=long_text, type="text", bboxes=[])

    # Split the document
    result = text_splitter.split(document, 0)

    # Verify the result
    assert result.path == "test.txt"
    assert result.exception == ""
    assert len(result.chunks) > 1
    assert result.processing_time > 0
    assert result.page == 0

    # Verify all chunks are strings
    assert all(isinstance(chunk, str) for chunk in result.chunks)

    # Verify chunks are not empty
    assert all(len(chunk) > 0 for chunk in result.chunks)


def test_text_splitter_split_method_with_exception():
    """Test the split method handles exceptions properly."""
    text_splitter = TextSplitter()

    # Create a document with None text (should cause an exception)
    document = Document(path="test.txt", text=None, type="text", bboxes=[])

    # Split the document
    result = text_splitter.split(document, 0)

    # Verify the result contains exception information
    assert result.path == "test.txt"
    assert result.exception != ""
    assert len(result.chunks) == 0
    assert result.processing_time > 0


def test_text_splitter_pipeline(document_df):
    """Test TextSplitter in a PySpark pipeline."""
    # Initialize the TextSplitter stage
    text_splitter = TextSplitter(
        inputCol="document",
        outputCol="chunks",
        chunk_size=200,
    )

    # Create a pipeline with the TextSplitter stage
    pipeline = PipelineModel(stages=[text_splitter])

    result_df = pipeline.transform(document_df)

    # Cache the result for performance
    result = result_df.select("chunks").cache()

    # Collect the results
    data = result.collect()

    # Check that exceptions are empty
    assert all(row.chunks.exception == "" for row in data)

    # Assert that there is at least one result
    assert len(data) > 0

    # Assert that the 'chunks' field is present in the result
    assert hasattr(data[0], "chunks")

    # Verify the chunks are not empty
    for row in data:
        assert row.chunks.chunks is not None
        assert len(row.chunks.chunks) > 0
        assert row.chunks.path is not None
        assert row.chunks.processing_time > 0


def test_text_splitter_pipeline_pandas(document_df):
    """Test TextSplitter with partitionMap (pandas mode)."""
    # Initialize the TextSplitter stage
    text_splitter = TextSplitter(
        inputCol="document",
        outputCol="chunks",
        chunk_size=200,
        partitionMap=True,
    )

    # Create a pipeline with the TextSplitter stage
    pipeline = PipelineModel(stages=[text_splitter])

    result_df = pipeline.transform(document_df)

    # Cache the result for performance
    result = result_df.select("chunks").cache()

    # Collect the results
    data = result.collect()

    # Check that exceptions are empty
    assert all(row.chunks.exception == "" for row in data)

    # Assert that there is at least one result
    assert len(data) > 0

    # Assert that the 'chunks' field is present in the result
    assert hasattr(data[0], "chunks")

    # Verify the chunks are not empty
    for row in data:
        assert row.chunks.chunks is not None
        assert len(row.chunks.chunks) > 0
        assert row.chunks.path is not None
        assert row.chunks.processing_time > 0
