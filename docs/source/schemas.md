Data Schemas
============

## Overview

This section provides comprehensive documentation on data schemas used throughout ScaleDP. These structured schemas ensure type safety, consistency, and clarity across all transformers and pipelines. They serve as the foundation for document processing workflows.

ScaleDP uses dataclass-based schemas that seamlessly integrate with both Python and PySpark SQL for efficient distributed processing.

## Core Schemas

* [**Document**](./schemas/document.md) - Represents a document with text, metadata, and layout information
* [**TextChunks**](./schemas/text_chunks.md) - Represents the output of text splitting operations
* [**Box**](./schemas/box.md) - Represents bounding boxes for spatial information

## Quick Reference

| Schema     | Purpose                              | Used By              |
|------------|--------------------------------------|----------------------|
| Document  | Input document representation        | Text Splitters, NER  |
| TextChunks| Split text chunks with metadata      | Embeddings, Search   |
| Box       | Bounding box and spatial info        | OCR, Layout Analysis |

## Quick Start

### Creating a Document

```python
from scaledp.schemas.Document import Document

doc = Document(
    path="/path/to/file.txt",
    text="Document content here...",
    type="text",
    bboxes=[]
)
```


### Using Bounding Boxes

```python
from scaledp.schemas.Box import Box

box = Box(
    text="Text content",
    score=0.95,
    x=10, y=20,
    width=100, height=50,
    angle=0.0
)
```

## Processing Pipeline

```
Document (Input)
    ↓
Text Splitter
    ↓
TextChunks (Split Results)
    ↓
Embeddings / NER / Other Processing
    ↓
Results
```

## Schema Integration

All schemas provide seamless integration with:

- **Python** - Native dataclasses with type hints
- **PySpark SQL** - `.get_schema()` method for SQL schema
- **DataFrames** - Direct use in Spark DataFrames
- **Pipelines** - Compatible with PySpark ML pipelines

## Best Practices

1. **Always provide path information** - Use meaningful file paths for tracking
2. **Set appropriate document types** - Use specific types (pdf, text, image)
3. **Include bounding boxes** - When layout information is available
4. **Track processing errors** - Capture exceptions in schema fields

For detailed information on each schema, see:

* [**Document Schema Documentation**](./schemas/document.md)
* [**TextChunks Schema Documentation**](./schemas/text_chunks.md)
* [**Box Schema Documentation**](./schemas/box.md)
* [**Complete Schemas Overview**](./schemas/index.md)

## Related Sections

* [**Text Splitters**](./splitters.md) - Document splitting transformers
* [**Embeddings**](./embeddings.md) - Text embedding generation
* [**Image Processing**](./image_processing.md) - Image document processing
* [**PDF Processing**](./pdf_processing.md) - PDF document handling
