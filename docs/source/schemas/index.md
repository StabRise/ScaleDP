(schemas)=
# Data Schemas

ScaleDP uses structured data schemas to represent documents, chunks, and spatial information throughout the processing pipeline. These schemas provide type safety, consistency, and clarity across all transformers.

## Core Schemas

### [Document](./document.md)
Represents a document with text content, file path, and metadata.

**Key Fields:**
- `path` - File path or document identifier
- `text` - Text content
- `type` - Document type (text, pdf, image)
- `bboxes` - List of bounding boxes
- `exception` - Error message (optional)

**Usage:** Input format for text splitters, embeddings, and NER models.

### [TextChunks](./text_chunks.md)
Represents the output of text splitting operations.

**Key Fields:**
- `path` - Original document path
- `chunks` - List of text chunks
- `exception` - Error message if any
- `processing_time` - Processing duration

**Usage:** Output from text splitters, input for embeddings.

### [Box](./box.md)
Represents a bounding box with position and text information.

**Key Fields:**
- `text` - Text content or label
- `score` - Confidence score
- `x`, `y` - Top-left coordinates
- `width`, `height` - Dimensions
- `angle` - Rotation angle

**Usage:** Layout information in documents, OCR results.

## Schema Hierarchy

```
Document
├── path: str
├── text: str
├── type: str
├── bboxes: list[Box]
│   ├── text: str
│   ├── score: float
│   ├── x: int
│   ├── y: int
│   ├── width: int
│   ├── height: int
│   └── angle: float
└── exception: str

TextChunks
├── path: str
├── chunks: list[str]
├── exception: str
└── processing_time: float
```

## Processing Pipeline Overview

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

## Quick Reference

| Schema      | Purpose                      | Input To           | Output From    |
|-------------|------------------------------|-------------------|-----------------|
| Document   | Represents document data     | Text Splitter, NER | PDF Reader     |
| TextChunks | Represents text chunks       | Embeddings, Search | Text Splitter  |
| Box        | Represents spatial region    | Layout Analysis    | OCR, Detector  |

## Common Operations

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

### Splitting Documents

```python
from scaledp.models.splitters.TextSplitter import TextSplitter

splitter = TextSplitter(chunk_size=500, chunk_overlap=50)
result_df = splitter.transform(df)
# Output: DataFrame with TextChunks
```

### Processing Chunks

```python
from pyspark.sql.functions import explode

# Flatten chunks for further processing
flattened = result_df.select(
    "chunks.path",
    explode("chunks.chunks").alias("chunk")
)
```

### Analyzing Box Operations

```python
from scaledp.schemas.Box import Box

# Calculate IoU between boxes
iou = Box.iou(box1, box2)

# Merge overlapping boxes
merged = Box.merge_overlapping_boxes(boxes, iou_threshold=0.3)

# Check if rotated
if box.is_rotated():
    handle_rotated_box(box)
```

## PySpark Schema Access

All schemas provide `.get_schema()` method for PySpark integration:

```python
from scaledp.schemas.Document import Document
from scaledp.schemas.TextChunks import TextChunks
from scaledp.schemas.Box import Box

# Get PySpark SQL schema
doc_schema = Document.get_schema()
chunks_schema = TextChunks.get_schema()
box_schema = Box.get_schema()

# Use in DataFrames
df = spark.createDataFrame(data, schema=doc_schema)
```

## Error Handling

All schemas support error tracking:

```python
from scaledp.schemas.Document import Document
from scaledp.schemas.TextChunks import TextChunks

# Document with error
doc = Document(
    path="file.txt",
    text="",
    type="text",
    bboxes=[],
    exception="Failed to read file"
)

# TextChunks with error
result = TextChunks(
    path="file.txt",
    chunks=None,
    exception="Splitting failed: Invalid input",
    processing_time=0.0
)

# Check for errors in pipeline
failed = result_df.filter("chunks.exception != ''")
```

## Best Practices

### 1. Always Provide Path Information
```python
# Good: includes path for tracking
doc = Document(path="document.pdf", ...)

# Avoid: missing path
doc = Document(path="", ...)
```

### 2. Set Document Types Correctly
```python
# Good: specific types
Document(..., type="pdf", ...)
Document(..., type="image", ...)
Document(..., type="text", ...)

# Avoid: generic types
Document(..., type="document", ...)
```

### 3. Include Bounding Boxes When Available
```python
# Good: includes layout info
doc = Document(
    path="page.jpg",
    text=text,
    type="image",
    bboxes=detected_boxes
)

# Acceptable: empty for plain text
doc = Document(
    path="text.txt",
    text=text,
    type="text",
    bboxes=[]
)
```

### 4. Track Processing Errors
```python
# Good: capture and report errors
try:
    result = process(doc)
except Exception as e:
    result = TextChunks(
        path=doc.path,
        chunks=None,
        exception=str(e),
        processing_time=0.0
    )
```

## Integration Examples

### Text Processing Pipeline
```python
from pyspark.ml import Pipeline
from scaledp.models.splitters.TextSplitter import TextSplitter
from scaledp.models.embeddings.HFEmbeddings import HFEmbeddings

# Create pipeline with schemas
stages = [
    TextSplitter(inputCol="document", outputCol="chunks"),
    HFEmbeddings(inputCol="chunks.chunks", outputCol="embeddings"),
]

pipeline = Pipeline(stages=stages)
result = pipeline.fit(df).transform(df)
```

### Batch Processing
```python
from scaledp.schemas.Document import Document

# Create documents from files
documents = []
for file_path in files:
    doc = Document(
        path=file_path,
        text=read_file(file_path),
        type="text",
        bboxes=[]
    )
    documents.append(doc)

# Process as DataFrame
df = spark.createDataFrame(
    [(doc,) for doc in documents],
    ["document"]
)
```

## See Also

- [Document Schema](./document.md) - Detailed documentation
- [TextChunks Schema](./text_chunks.md) - Detailed documentation
- [Box Schema](./box.md) - Detailed documentation
- [Text Splitters](../models/splitters/index.md)
- [Embeddings](../models/embeddings.md)
