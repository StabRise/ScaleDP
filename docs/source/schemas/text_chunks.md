(TextChunks)=
# TextChunks Schema

## Overview

`TextChunks` is a structured data schema that represents the output of text splitting operations. It contains the split text chunks along with metadata about the original document and processing information. This schema is produced by text splitter transformers and serves as input for downstream processing tasks like embedding generation.

The TextChunks schema maintains traceability by preserving the document path and captures processing details for debugging and monitoring.

## Schema Structure

```python
from scaledp.schemas.TextChunks import TextChunks
from typing import Optional

# TextChunks dataclass definition
@dataclass(order=True)
class TextChunks:
    path: Optional[str]              # Original document path
    chunks: Optional[list[str]]      # List of text chunks
    exception: Optional[str] = ""    # Error message if any
    processing_time: Optional[float] = 0.0  # Processing duration in seconds
```

## Fields

| Field            | Type              | Required | Description                                    |
|------------------|-------------------|----------|------------------------------------------------|
| path             | Optional[str]     | No       | Original document file path                    |
| chunks           | Optional[list[str]]| No       | List of text chunks from splitting             |
| exception        | Optional[str]     | No       | Error message if splitting failed              |
| processing_time  | Optional[float]   | No       | Time taken to split document (in seconds)      |

## Usage Examples

### Basic TextChunks Creation

```python
from scaledp.schemas.TextChunks import TextChunks

# Successful split result
result = TextChunks(
    path="/documents/file.txt",
    chunks=[
        "First chunk of text...",
        "Second chunk of text...",
        "Third chunk of text..."
    ],
    exception="",
    processing_time=0.125
)
```


## PySpark Schema

```python
from scaledp.schemas.TextChunks import TextChunks

schema = TextChunks.get_schema()
# StructType([
#     StructField('path', StringType(), True),
#     StructField('chunks', ArrayType(StringType()), True),
#     StructField('exception', StringType(), True),
#     StructField('processing_time', DoubleType(), True)
# ])
```

## Processing Pipeline Integration

### With Text Splitter

```python
from scaledp.models.splitters.TextSplitter import TextSplitter

# TextSplitter produces TextChunks output
text_splitter = TextSplitter(
    inputCol="document",
    outputCol="chunks",  # Output is TextChunks schema
    chunk_size=500,
)

result_df = text_splitter.transform(input_df)
# result_df has column "chunks" of type TextChunks
```

### Flattening Chunks

```python
from pyspark.sql.functions import explode

# Flatten chunks for further processing
flattened = result_df.select(
    "chunks.path",
    explode("chunks.chunks").alias("chunk"),
    "chunks.processing_time"
)
```

### Chunk Distribution

```python
from pyspark.sql.functions import size

# Analyze chunk distribution
result_df.select(
    "chunks.path",
    size("chunks.chunks").alias("num_chunks"),
).groupBy().agg({"num_chunks": "avg"}).show()
```

## Related Schemas

- [Document](./document.md) - Input schema for text splitting
- [Box](./box.md) - Bounding box schema

## See Also

- [Text Splitters](../models/splitters/index.md)
- [BaseTextSplitter](../models/splitters/base_text_splitter.md)
- [TextSplitter](../models/splitters/text_splitter.md)
- [Embeddings](../models/embeddings.md)
