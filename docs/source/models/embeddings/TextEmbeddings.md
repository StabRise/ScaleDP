(TextEmbeddings)=
# TextEmbeddings

## Overview

`TextEmbeddings` is a text embedding transformer based on the SentenceTransformer model. It is designed to efficiently generate embeddings for text data using a pre-trained model. The transformer is implemented as a PySpark ML transformer and can be integrated into Spark pipelines for scalable text embedding tasks.

The transformer supports two input types:
- **Raw text**: Single text string per row
- **TextChunks schema**: Structured input with path, chunks (list of strings), exception, and processing_time fields

## Usage Examples

### Raw Text Input

```python
from scaledp import TextEmbeddings, PipelineModel

text_embeddings = TextEmbeddings(
    inputCol="text",
    outputCol="embeddings",
    keepInputData=True,
    model="all-MiniLM-L6-v2",
    batchSize=1,
    device="cpu",
)

# Transform the text dataframe through the embedding stage
pipeline = PipelineModel(stages=[text_embeddings])
result = pipeline.transform(text_df)
result.show()
```

### TextChunks Schema Input

```python
from scaledp import TextEmbeddings, PipelineModel
from scaledp.schemas.TextChunks import TextChunks

# Assuming df has a column with TextChunks schema
text_embeddings = TextEmbeddings(
    inputCol="text_chunks",
    outputCol="embeddings",
    keepInputData=True,
    model="all-MiniLM-L6-v2",
    batchSize=2,
    device="cpu",
)

# Transform the dataframe - automatically detects TextChunks schema
pipeline = PipelineModel(stages=[text_embeddings])
result = pipeline.transform(df)

# Each chunk in the TextChunks list generates one row with preserved metadata
# Embeddings include: path, data (embedding vector), type, exception, processing_time
result.show()
```

## Parameters

| Parameter         | Type    | Description                                      | Default                     |
|-------------------|---------|--------------------------------------------------|-----------------------------|
| inputCol          | str     | Input text column                                | "text"                    |
| outputCol         | str     | Output column for embeddings                     | "embeddings"              |
| keepInputData     | bool    | Keep input data in output                        | True                        |
| model             | str     | Pre-trained model identifier                     | "all-MiniLM-L6-v2"        |
| batchSize         | int     | Batch size for inference                         | 1                           |
| device            | Device  | Inference device (CPU/GPU)                       | Device.CPU                  |
| numPartitions     | int     | Number of partitions                             | 1                           |
| partitionMap      | bool    | Use partitioned mapping                          | False                       |
| pageCol           | str     | Page column                                      | "page"                    |
| pathCol           | str     | Path column                                      | "path"                    |

## Input and Output Schemas

### Input Types

The transformer automatically detects and handles two input types:

#### Raw Text
- **Type Detection**: StringType column
- **Behavior**: Each row generates one embedding
- **Output Type Field**: "text"

#### TextChunks Schema
- **Type Detection**: StructType matching TextChunks schema
- **Behavior**: Each chunk generates one embedding row (using explode internally)
- **Output Type Field**: "text_chunk"
- **Metadata Preservation**: path, exception, processing_time preserved in output

### Output Schema

All outputs use the `EmbeddingsOutput` schema:

| Field            | Type          | Description                                          |
|------------------|---------------|------------------------------------------------------|
| path             | Optional[str] | Source path (from TextChunks or "memory" for text)   |
| data             | Optional[list[float]] | Embedding vector                             |
| type             | Optional[str] | "text" or "text_chunk"                               |
| exception        | Optional[str] | Error message if generation failed                   |
| processing_time  | Optional[float] | Processing duration in seconds                      |

## Behavior

### Raw Text Input
```
Input:  1 row × 1 column
        "This is a text sample"

Transform →

Output: 1 row × 1 column
        EmbeddingsOutput(
            path="memory",
            data=[0.123, -0.456, ...],
            type="text",
            exception="",
            processing_time=0.045
        )
```

### TextChunks Input
```
Input:  1 row × 1 column
        TextChunks(
            path="doc.txt",
            chunks=["chunk 1", "chunk 2", "chunk 3"],
            exception="",
            processing_time=1.5
        )

Transform →

Output: 3 rows × 1 column (one per chunk)
        Row 1: EmbeddingsOutput(path="doc.txt", data=[...], type="text_chunk", ...)
        Row 2: EmbeddingsOutput(path="doc.txt", data=[...], type="text_chunk", ...)
        Row 3: EmbeddingsOutput(path="doc.txt", data=[...], type="text_chunk", ...)
```

## Schema Detection

The transformer automatically detects input schema using `_is_text_chunks_column()`:
- Checks if column type is StructType
- Compares against TextChunks.get_schema()
- Falls back to raw text handling if not TextChunks

## Performance Considerations

### Batch Size
- Affects memory usage and inference speed
- Larger batches are faster but require more GPU memory
- Default: 1 (conservative)
- Recommended: 16-64 for GPUs, 2-8 for CPUs

### Partition Map
- `partitionMap=False` (default): Uses regular UDF
- `partitionMap=True`: Uses pandas_udf for better performance
- Recommended: Enable for large datasets or when numPartitions > 1

### Parallelization
- Set `numPartitions` > 1 to distribute computation
- Pairs well with `partitionMap=True` for optimal performance
- Consider your cluster size and model size

## Notes
- The transformer uses the SentenceTransformer model for generating text embeddings.
- Supports batch processing and distributed inference with Spark.
- Automatically detects input schema type (raw text vs TextChunks).
- When `partitionMap=True`, uses pandas_udf for better performance on large datasets.
- Additional parameters can be set using the corresponding setter methods.
- See [EmbeddingsOutput Schema Documentation](../../schemas/embeddings_output.md) for detailed output schema information.
- See [TextChunks Schema Documentation](../../schemas/text_chunks.md) for detailed input schema information.
