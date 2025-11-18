(TextSplitter)=
# TextSplitter

## Overview

`TextSplitter` is a concrete implementation of [`BaseTextSplitter`](./base_text_splitter.md) that uses semantic text splitting to divide documents into meaningful chunks. It splits text based on content boundaries while respecting chunk size and overlap constraints. The splitter operates on **Document struct columns** and preserves document metadata through the splitting process.

## Inheritance

- Inherits from [`BaseTextSplitter`](./base_text_splitter.md), which provides the core text splitting framework.
- Uses `semantic_text_splitter` library for intelligent chunking based on content semantics.

## Key Features

- **Semantic Splitting**: Intelligently splits text at natural content boundaries (sentences, paragraphs)
- **Configurable Chunk Size**: Control the maximum size of each chunk
- **Overlap Support**: Maintain context between chunks with configurable overlap
- **Metadata Preservation**: Keeps document path and type information throughout splitting
- **Batch Processing**: Supports distributed processing across Spark cluster
- **Error Handling**: Gracefully handles malformed or problematic text

## Usage Example

```python
from scaledp.models.splitters.TextSplitter import TextSplitter
from pyspark.ml import PipelineModel

# Create a text splitter
text_splitter = TextSplitter(
    inputCol="document",        # Input Document struct column
    outputCol="chunks",         # Output TextChunks column
    chunk_size=500,             # Max chunk size in characters
    chunk_overlap=50,           # Character overlap between chunks
    keepInputData=True,         # Keep original document in output
)

# Use in a Spark pipeline
pipeline = PipelineModel(stages=[text_splitter])
result_df = pipeline.transform(input_df)

# Inspect results
for row in result_df.collect():
    print(f"Document: {row.chunks.path}")
    print(f"Number of chunks: {len(row.chunks.chunks)}")
    print(f"Processing time: {row.chunks.processing_time}s")
    if row.chunks.exception:
        print(f"Error: {row.chunks.exception}")
```

## Parameters

All parameters from [`BaseTextSplitter`](./base_text_splitter.md) are inherited. No additional parameters specific to `TextSplitter`.

| Parameter         | Type    | Description                                      | Default                     |
|-------------------|---------|--------------------------------------------------|-----------------------------|
| inputCol          | str     | Input Document struct column                     | "document"                  |
| outputCol         | str     | Output column for TextChunks results             | "chunks"                    |
| keepInputData     | bool    | Keep input document column in output             | True                        |
| chunk_size        | int     | Size of each chunk in characters                 | 500                         |
| chunk_overlap     | int     | Number of characters to overlap between chunks   | 0                           |
| numPartitions     | int     | Number of partitions for coalescing              | 1                           |
| partitionMap      | bool    | Use pandas_udf for distributed processing        | False                       |

## Input Schema

Requires a DataFrame column containing **Document struct**. For detailed schema information and examples, see [Document Schema Documentation](../../schemas/document.md).

```python
from scaledp.schemas.Document import Document

# Document struct fields
Document(
    path: str,              # File path or identifier
    text: str,              # Text content to split
    type: str,              # Document type (e.g., "text", "pdf")
    bboxes: list[Box],      # Bounding boxes (empty for plain text)
    exception: str = ""     # Optional error message
)
```

## Output Schema

Produces output column with **TextChunks struct**. For detailed schema information and examples, see [TextChunks Schema Documentation](../../schemas/text_chunks.md).

```python
from scaledp.schemas.TextChunks import TextChunks

# TextChunks struct fields
TextChunks(
    path: str,                   # Original document path
    chunks: list[str],           # List of text chunks
    exception: str = "",         # Error message if any
    processing_time: float = 0.0 # Splitting duration in seconds
)
```

## Examples

### Example 1: Basic Text Splitting

```python
from scaledp.schemas.Document import Document
from scaledp.models.splitters.TextSplitter import TextSplitter

# Create a test document
long_text = "This is a long document. " * 100
doc = Document(
    path="test.txt",
    text=long_text,
    type="text",
    bboxes=[]
)

# Create splitter
splitter = TextSplitter(chunk_size=200, chunk_overlap=20)

# Split the document
result = splitter.split(doc)
print(f"Created {len(result.chunks)} chunks")
print(f"Processing time: {result.processing_time}ms")
```

### Example 2: DataFrame Transformation

```python
from pyspark.sql.functions import col, lit, struct
from scaledp.models.splitters.TextSplitter import TextSplitter
from scaledp.schemas.Document import Document

# Create a DataFrame with Document structs
df = spark.createDataFrame([
    {
        "document": {
            "path": "file1.txt",
            "text": "First document content...",
            "type": "text",
            "bboxes": [],
            "exception": ""
        }
    }
])

# Apply text splitter
splitter = TextSplitter(chunk_size=500, chunk_overlap=50)
result_df = splitter.transform(df)

# View results
result_df.select("document.path", "chunks.chunks").show()
```

### Example 3: Pipeline Integration

```python
from pyspark.ml import Pipeline
from scaledp.models.splitters.TextSplitter import TextSplitter

# Create a text splitter stage for a pipeline
text_splitter = TextSplitter(
    inputCol="document",
    outputCol="chunks",
    chunk_size=500,
    chunk_overlap=50,
)

# Create and fit pipeline
pipeline = Pipeline(stages=[text_splitter])
model = pipeline.fit(input_df)

# Transform data
output_df = model.transform(input_df)
```

## Performance Considerations

- **Chunk Size**: Larger chunks process faster but may be too large for downstream tasks
- **Chunk Overlap**: Increases output size but helps maintain context between chunks
- **Batch Mode (partitionMap=False)**: Suitable for small documents or single-threaded processing
- **Pandas Mode (partitionMap=True)**: Better for large-scale distributed processing
- **numPartitions**: For pandas mode, controls how many partitions to coalesce to

## Error Handling

The splitter handles various error cases gracefully:

```python
# Handling errors in output
for row in result_df.collect():
    if row.chunks.exception:
        print(f"Error processing {row.chunks.path}: {row.chunks.exception}")
    else:
        print(f"Successfully split into {len(row.chunks.chunks)} chunks")
```

## Notes

- Semantic splitting provides better quality chunks than simple character-based splitting
- Document path is preserved in output for tracking and debugging
- Processing time is recorded for performance monitoring
- Empty documents are handled gracefully and return an empty chunk list

## Related Classes

- [`BaseTextSplitter`](./base_text_splitter.md) - Abstract base class for text splitters
- [`BaseSplitter`](./base_splitter.md) - Base transformer for all splitter implementations
- [`Document`](../../schemas/document.md) - Input schema for documents
- [`TextChunks`](../../schemas/text_chunks.md) - Output schema for text chunks
- [`Box`](../../schemas/box.md) - Bounding box schema
