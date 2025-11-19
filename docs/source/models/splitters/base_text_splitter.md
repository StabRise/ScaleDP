(BaseTextSplitter)=
# BaseTextSplitter

## Overview

`BaseTextSplitter` is an abstract base class for text splitting transformers in PySpark. It provides common functionality for splitting documents into chunks while preserving metadata like file paths and document types. It is designed for extensibility and serves as the foundation for concrete text splitting implementations like [`TextSplitter`](./text_splitter.md).

The splitter operates on **Document struct columns**, which contain structured data including text content, file path, document type, and bounding boxes.

## Inheritance

- Inherits from [`BaseSplitter`](./base_splitter.md), which provides core Spark ML transformer functionality and schema handling.
- Mixes in `HasColumnValidator` and `HasDefaultEnum` for validation and enumeration support.
- Extends `DefaultParamsReadable` and `DefaultParamsWritable` for serialization support.

## Key Features

- **Document-Centric**: Works with Document struct columns containing path, text, type, and bboxes
- **Metadata Preservation**: Maintains document metadata (path, document type) through the splitting process
- **Flexible Chunking**: Configurable chunk size and overlap for text splitting
- **Distributed Processing**: Supports both regular UDF and pandas_udf (partitionMap) modes for Spark batch processing
- **Error Handling**: Captures and reports processing exceptions in output

## Usage Example

```python
from scaledp.models.splitters.TextSplitter import TextSplitter
from scaledp.schemas.Document import Document

# Create a splitter with custom parameters
splitter = TextSplitter(
    inputCol="document",      # Column containing Document structs
    outputCol="chunks",       # Output column for TextChunks
    chunk_size=500,           # Characters per chunk
    chunk_overlap=50,         # Character overlap between chunks
)

# Use in a Spark pipeline
result_df = splitter.transform(input_df)
```

## Parameters

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

The input column must contain **Document struct**. For detailed schema information, see [Document Schema Documentation](../../schemas/document.md).

**Key Fields:**
- `path` - File path or document identifier
- `text` - Text content to split
- `type` - Document type (e.g., "text", "pdf")
- `bboxes` - Bounding boxes (empty for text documents)
- `exception` - Error message if any (optional)

## Output Schema

The output column contains **TextChunks struct**. For detailed schema information, see [TextChunks Schema Documentation](../../schemas/text_chunks.md).

**Key Fields:**
- `path` - Original document path
- `chunks` - List of text chunks
- `exception` - Error message if splitting failed
- `processing_time` - Time taken to split document (seconds)

## Notes

- The splitter is abstract and cannot be instantiated directly. Use concrete implementations like `TextSplitter`.
- Input documents must contain text and path information in the Document struct format.
- Chunk overlap can help maintain context between chunks for semantic meaning.
- The `partitionMap` option enables pandas_udf mode for better performance on large datasets but requires careful configuration.
- All errors during splitting are captured and reported in the `exception` field of the output.

## Related Classes

- [`TextSplitter`](./text_splitter.md) - Concrete implementation using semantic text splitting
- [`BaseSplitter`](./base_splitter.md) - Base transformer for all splitter implementations
- [`Document`](../../schemas/document.md) - Input schema class
- [`TextChunks`](../../schemas/text_chunks.md) - Output schema class
- [`Box`](../../schemas/box.md) - Bounding box schema class
