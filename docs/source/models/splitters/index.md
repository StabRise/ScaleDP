(splitters)=
# Text Splitters

Text splitters are transformers that divide documents into smaller, manageable chunks while preserving metadata and handling errors gracefully. They are essential for processing large documents in downstream tasks like embedding generation or information extraction.

## Overview

ScaleDP provides a flexible text splitting framework built on PySpark ML pipelines:

- **Semantic Splitting**: Intelligently split text at natural boundaries
- **Distributed Processing**: Scale to large datasets with Spark
- **Metadata Preservation**: Keep document information through splitting
- **Error Handling**: Graceful error reporting and recovery
- **Pipeline Integration**: Seamless integration with PySpark ML pipelines

## Available Splitters

### [TextSplitter](./text_splitter.md)
Main text splitter implementation using semantic splitting. Intelligently divides text based on content boundaries while respecting chunk size and overlap constraints.

**Key Features:**
- Semantic text splitting
- Configurable chunk size and overlap
- Support for distributed processing
- Metadata preservation

## Base Classes

### [BaseTextSplitter](./base_text_splitter.md)
Abstract base class for text splitters. Provides common functionality for operating on Document struct columns and producing TextChunks outputs.

### [BaseSplitter](./base_splitter.md)
Foundation class for all splitter transformers. Extends PySpark's Transformer and defines the common interface for all splitter implementations.

## Quick Start

```python
from scaledp.models.splitters.TextSplitter import TextSplitter
from scaledp.schemas.Document import Document

# Create a splitter
splitter = TextSplitter(
    inputCol="document",       # Input Document struct column
    outputCol="chunks",        # Output TextChunks column
    chunk_size=500,            # Max chunk size
    chunk_overlap=50,          # Character overlap
)

# Use in a Spark pipeline
result_df = splitter.transform(input_df)

# Access results
for row in result_df.collect():
    print(f"Path: {row.chunks.path}")
    print(f"Chunks: {len(row.chunks.chunks)}")
    print(f"Time: {row.chunks.processing_time}s")
```

## Input/Output Schema

### Input: [Document Struct](../../schemas/document.md)

| Field      | Type              | Description                   |
|------------|-------------------|-------------------------------|
| path       | string            | Document identifier/file path  |
| text       | string            | Text content to split         |
| type       | string            | Document type                 |
| bboxes     | array<Box>        | Bounding boxes (optional)     |
| exception  | string            | Error message (optional)      |

For detailed information, see [Document Schema Documentation](../../schemas/document.md).

### Output: [TextChunks Struct](../../schemas/text_chunks.md)

| Field            | Type          | Description              |
|------------------|---------------|--------------------------|
| path             | string        | Original document path   |
| chunks           | array<string> | List of text chunks      |
| exception        | string        | Error message if any     |
| processing_time  | double        | Processing duration (sec)|

For detailed information, see [TextChunks Schema Documentation](../../schemas/text_chunks.md).

## Use Cases

### Document Processing Pipeline
```python
from pyspark.ml import Pipeline

pipeline = Pipeline(stages=[
    text_cleaner,          # Clean input text
    text_splitter,         # Split into chunks
    embedding_generator,   # Generate embeddings
])

result = pipeline.fit(data).transform(data)
```

### Large Document Handling
```python
# Split large documents for processing
splitter = TextSplitter(
    chunk_size=1000,       # Larger chunks
    chunk_overlap=200,     # More context
)
```

### Batch Processing
```python
# Process multiple documents
result_df = splitter.transform(multi_doc_df)

# Filter errors
valid_results = result_df.filter("chunks.exception == ''")
```

## Performance Tuning

### Chunk Size Selection
- **Smaller chunks** (100-300): Better for semantic search, more processing overhead
- **Medium chunks** (500-1000): Good balance for most use cases
- **Large chunks** (2000+): Faster processing, less granular results

### Overlap Configuration
- **No overlap** (0): Fastest processing, potential context loss
- **Small overlap** (10-50): Balanced approach for most cases
- **Large overlap** (200+): Maximum context preservation

### Distributed Processing
```python
# Use pandas_udf mode for large datasets
splitter = TextSplitter(
    partitionMap=True,     # Enable pandas_udf mode
    numPartitions=100,     # Set target partitions
)
```

## Extending Splitters

Create custom splitters by inheriting from `BaseTextSplitter`:

```python
from scaledp.models.splitters.BaseTextSplitter import BaseTextSplitter
from scaledp.schemas.Document import Document
from scaledp.schemas.TextChunks import TextChunks

class MyCustomSplitter(BaseTextSplitter):
    def split(self, document: Document) -> TextChunks:
        # Implement custom splitting logic
        chunks = my_splitting_algorithm(document.text)
        return TextChunks(
            path=document.path,
            chunks=chunks,
            exception="",
            processing_time=0.0
        )
```

## Error Handling

All splitters handle errors gracefully:

```python
result_df = splitter.transform(df)

# Check for errors
error_df = result_df.filter("chunks.exception != ''")
print(f"Failed: {error_df.count()}")

# Process successful results
success_df = result_df.filter("chunks.exception == ''")
```

## Documentation Index

- [BaseSplitter](./base_splitter.md) - Foundation class
- [BaseTextSplitter](./base_text_splitter.md) - Text splitting base class
- [TextSplitter](./text_splitter.md) - Semantic text splitter implementation

## Related Resources

- [Document Schema](../../schemas/document.md) - Input document structure
- [TextChunks Schema](../../schemas/text_chunks.md) - Output chunks structure
- [Box Schema](../../schemas/box.md) - Bounding box structure
- [All Data Schemas](../../schemas/index.md) - Complete schema documentation
- [PySpark ML Pipelines](https://spark.apache.org/docs/latest/ml-pipeline.html)
- [Transformer API](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.Transformer.html)
