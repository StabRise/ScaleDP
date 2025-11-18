Text Splitters
==============

## Overview

This section provides comprehensive documentation on text splitters in ScaleDP. Text splitters divide documents into smaller, manageable chunks while preserving metadata and handling errors gracefully. They are essential for processing large documents in downstream tasks like embedding generation or information extraction.

ScaleDP provides a flexible text splitting framework built on PySpark ML pipelines with support for semantic splitting, distributed processing, and metadata preservation.

## Text Splitters

* [**TextSplitter**](./models/splitters/text_splitter.md) - Semantic text splitter using intelligent chunking

## Base Classes

* [**BaseTextSplitter**](./models/splitters/base_text_splitter.md) - Abstract base for text splitters
* [**BaseSplitter**](./models/splitters/base_splitter.md) - Foundation class for all splitter implementations

## Quick Start

```python
from scaledp.models.splitters.TextSplitter import TextSplitter

# Create a text splitter
splitter = TextSplitter(
    inputCol="document",
    outputCol="chunks",
    chunk_size=500,
    chunk_overlap=50,
)

# Use in a Spark pipeline
result_df = splitter.transform(input_df)
```

## Related Schemas

* [**Data Schemas**](./schemas.md) - Complete schema documentation
  * [**Document**](./schemas/document.md) - Input document schema
  * [**TextChunks**](./schemas/text_chunks.md) - Output chunks schema
  * [**Box**](./schemas/box.md) - Bounding box schema

## Features

- **Semantic Splitting** - Intelligently split text at natural boundaries
- **Distributed Processing** - Scale to large datasets with Spark
- **Metadata Preservation** - Keep document information through splitting
- **Error Handling** - Graceful error reporting and recovery
- **Pipeline Integration** - Seamless integration with PySpark ML pipelines

For detailed information and examples, see the [Text Splitters Documentation](./models/splitters/index.md).
