(Document)=
# Document Schema

## Overview

`Document` is a structured data schema that represents a document with its text content, metadata, and structural information. It serves as the standard input format for text processing transformers in ScaleDP, including text splitters, embeddings, and NER models.

The Document schema contains all necessary information for processing a document through a pipeline: the text content, file path, document type, and bounding box information for layout-aware processing.

## Schema Structure

```python
from scaledp.schemas.Document import Document
from scaledp.schemas.Box import Box

# Document dataclass definition
@dataclass
class Document:
    path: str              # File path or document identifier
    text: str              # Text content
    type: str              # Document type (e.g., "text", "pdf", "image")
    bboxes: list[Box]      # List of bounding boxes
    exception: str = ""    # Error message (optional)
```

## Fields

| Field     | Type         | Required | Description                                          |
|-----------|--------------|----------|------------------------------------------------------|
| path      | str          | Yes      | File path or unique document identifier              |
| text      | str          | Yes      | Text content of the document                         |
| type      | str          | Yes      | Document type (e.g., "text", "pdf", "image")        |
| bboxes    | list[Box]    | No       | List of bounding boxes for text regions              |
| exception | str          | No       | Error message if document processing failed          |

## Related Schemas

### Box
The `Box` schema represents a bounding box with position and size information:

```python
@dataclass
class Box:
    text: str          # Text content of the box
    score: float       # Confidence score
    x: int             # X coordinate
    y: int             # Y coordinate
    width: int         # Box width
    height: int        # Box height
    angle: float = 0.0 # Rotation angle (degrees)
```

## Usage Examples

### Creating a Document from Text

```python
from scaledp.schemas.Document import Document

# Create a simple text document
doc = Document(
    path="test.txt",
    text="This is a sample document.",
    type="text",
    bboxes=[]
)
```

## PySpark Schema

When converting to PySpark SQL schema:

```python
from scaledp.schemas.Document import Document

schema = Document.get_schema()
# StructType([
#     StructField('path', StringType(), True),
#     StructField('text', StringType(), True),
#     StructField('type', StringType(), True),
#     StructField('bboxes', ArrayType(BoxType(...)), True),
#     StructField('exception', StringType(), True)
# ])
```

## Related Schemas

- [Box](./box.md) - Bounding box schema
- [TextChunks](./text_chunks.md) - Output schema for text splitters

## See Also

- [Text Splitters](../models/splitters/index.md)
- [Embeddings](../models/embeddings.md)
- [NER Models](../models/ner.md)
