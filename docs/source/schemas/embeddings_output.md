(EmbeddingsOutput)=
# EmbeddingsOutput Schema

## Overview

`EmbeddingsOutput` is a structured data schema that represents the output of text embedding operations. It contains the generated embedding vector along with metadata about the source document, processing information, and error tracking. This schema is produced by embedding transformers and serves as the standard output format for all embedding generation tasks.

The EmbeddingsOutput schema maintains traceability by preserving the source path and captures processing details for debugging, monitoring, and downstream analysis.

## Schema Structure

```python
from scaledp.schemas.EmbeddingsOutput import EmbeddingsOutput
from typing import Optional

# EmbeddingsOutput dataclass definition
@dataclass(order=True)
class EmbeddingsOutput:
    path: Optional[str]              # Source document or chunk path
    data: Optional[list[float]]      # Embedding vector
    type: Optional[str]              # Type of input ("text" or "text_chunk")
    exception: Optional[str] = ""    # Error message if any
    processing_time: Optional[float] = 0.0  # Processing duration in seconds
```

## Fields

| Field            | Type              | Required | Description                                    |
|------------------|-------------------|----------|------------------------------------------------|
| path             | Optional[str]     | No       | Source document file path or "memory"          |
| data             | Optional[list[float]]| No       | Embedding vector (list of floats)              |
| type             | Optional[str]     | No       | Input type: "text" or "text_chunk"             |
| exception        | Optional[str]     | No       | Error message if embedding generation failed   |
| processing_time  | Optional[float]   | No       | Time taken to generate embedding (in seconds)  |

## Usage Examples

### Basic EmbeddingsOutput Creation

```python
from scaledp.schemas.EmbeddingsOutput import EmbeddingsOutput

# Successful embedding result
result = EmbeddingsOutput(
    path="/documents/file.txt",
    data=[0.123, -0.456, 0.789, ...],  # Embedding vector
    type="text",
    exception="",
    processing_time=0.045
)

# Embedding from text chunks
chunk_result = EmbeddingsOutput(
    path="/documents/file.txt",
    data=[0.234, -0.567, 0.890, ...],
    type="text_chunk",
    exception="",
    processing_time=0.032
)
```

## PySpark Schema

```python
from scaledp.schemas.EmbeddingsOutput import EmbeddingsOutput

schema = EmbeddingsOutput.get_schema()
# StructType([
#     StructField('path', StringType(), True),
#     StructField('data', ArrayType(DoubleType()), True),
#     StructField('type', StringType(), True),
#     StructField('exception', StringType(), True),
#     StructField('processing_time', DoubleType(), True)
# ])
```

## Processing Pipeline Integration

### With TextEmbeddings - Raw Text Input

```python
from scaledp.models.embeddings.TextEmbeddings import TextEmbeddings

# TextEmbeddings produces EmbeddingsOutput from raw text
text_embeddings = TextEmbeddings(
    inputCol="text",
    outputCol="embeddings",  # Output is EmbeddingsOutput schema
    model="all-MiniLM-L6-v2",
)

result_df = text_embeddings.transform(input_df)
# result_df has column "embeddings" of type EmbeddingsOutput
# Each row has type="text"
```

### With TextEmbeddings - TextChunks Input

```python
from scaledp.models.splitters.TextSplitter import TextSplitter
from scaledp.models.embeddings.TextEmbeddings import TextEmbeddings

# Split text into chunks
splitter = TextSplitter(
    inputCol="document",
    outputCol="chunks",  # Output is TextChunks
    chunk_size=500,
)

# Generate embeddings from chunks
embeddings = TextEmbeddings(
    inputCol="chunks",
    outputCol="embeddings",  # Output is EmbeddingsOutput
    model="all-MiniLM-L6-v2",
)

# Create pipeline
from pyspark.ml import Pipeline
pipeline = Pipeline(stages=[splitter, embeddings])
result_df = pipeline.fit(input_df).transform(input_df)

# result_df has column "embeddings" of type EmbeddingsOutput
# Each chunk generates one embedding row with type="text_chunk"
# Path metadata is preserved from TextChunks
```

## Related Schemas

- [TextChunks](./text_chunks.md) - Input schema for chunk-based embeddings
- [Document](./document.md) - Input schema for raw text embeddings

## See Also

- [TextEmbeddings Transformer](../models/embeddings/TextEmbeddings.md)
- [Text Splitters](../models/splitters/index.md)
- [Embeddings](../models/embeddings.md)

## Best Practices

### 1. Always Check for Errors
```python
# Good: check exception field
successful = df.filter(df.embeddings.exception == "")

# Process only successful embeddings
vectors = successful.select("embeddings.data")
```

### 2. Preserve Path Information
```python
# Good: path is automatically preserved from TextChunks
# Use it for traceability
embeddings_with_path = result_df.select(
    "embeddings.path",
    "embeddings.data",
    "embeddings.type"
)
```

### 3. Monitor Processing Time
```python
# Good: track embedding generation performance
stats = result_df.select("embeddings.processing_time").describe()
stats.show()
```

### 4. Use Appropriate Batch Sizes
```python
# Good: adjust batch size for your dataset and hardware
embeddings = TextEmbeddings(
    inputCol="chunks",
    outputCol="embeddings",
    batchSize=32,  # Tune based on GPU memory
    model="all-MiniLM-L6-v2",
)
```

## Integration Examples

### Full Text Processing Pipeline

```python
from pyspark.ml import Pipeline
from scaledp.models.splitters.TextSplitter import TextSplitter
from scaledp.models.embeddings.TextEmbeddings import TextEmbeddings

# Create complete pipeline
stages = [
    TextSplitter(
        inputCol="document",
        outputCol="chunks",
        chunk_size=500,
    ),
    TextEmbeddings(
        inputCol="chunks",
        outputCol="embeddings",
        model="all-MiniLM-L6-v2",
        batchSize=32,
    ),
]

pipeline = Pipeline(stages=stages)
result_df = pipeline.fit(documents_df).transform(documents_df)

# Access embeddings
result_df.select(
    "embeddings.path",
    "embeddings.data",
    "embeddings.type",
    "embeddings.processing_time"
).show()
```

### Batch Embedding Generation

```python
from scaledp.models.embeddings.TextEmbeddings import TextEmbeddings

# Configure for large-scale processing
embeddings = TextEmbeddings(
    inputCol="text",
    outputCol="embeddings",
    model="all-MiniLM-L6-v2",
    batchSize=64,
    partitionMap=True,  # Use pandas_udf for distributed processing
    numPartitions=8,
)

# Process large dataset
result_df = embeddings.transform(input_df)

# Save embeddings for later use
result_df.write.mode("overwrite").parquet("s3://bucket/embeddings/")
```

### Embedding Storage and Retrieval

```python
from pyspark.sql.functions import col

# Save embeddings to vector database or storage
embeddings_storage = result_df.select(
    col("embeddings.path").alias("document_id"),
    col("embeddings.data").alias("vector"),
    col("embeddings.type").alias("embedding_type"),
)

embeddings_storage.write.format("parquet").mode("overwrite").save("embeddings/")

# Load and retrieve
loaded = spark.read.parquet("embeddings/")
loaded.filter(col("embedding_type") == "text_chunk").show()
```