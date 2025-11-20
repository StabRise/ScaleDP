Embeddings
==========

## Overview

This section provides an overview of the various embedding transformers available in ScaleDP for processing text and other data types. These transformers are designed to generate embeddings that can be used for tasks such as clustering, classification, and semantic similarity.

Embeddings are generated from text data using neural language models and produce high-dimensional vectors that capture semantic meaning. ScaleDP embeddings support both raw text and structured text chunks as input, with automatic schema detection.

## Text Embeddings

* [**TextEmbeddings**](models/embeddings/TextEmbeddings.md) - Generate embeddings from raw text or TextChunks

## Base Embeddings

* [**BaseEmbeddings**](models/embeddings/BaseEmbeddings.md) - Base class for embedding transformers

## Related Schemas

The embedding transformers work with the following data schemas:

* [**TextChunks Schema**](schemas/text_chunks.md) - Input schema for chunk-based embeddings (from text splitters)
* [**EmbeddingsOutput Schema**](schemas/embeddings_output.md) - Output schema containing embedding vectors and metadata

## Quick Start

### Embedding Raw Text

```python
from scaledp.models.embeddings.TextEmbeddings import TextEmbeddings

embeddings = TextEmbeddings(
    inputCol="text",
    outputCol="embeddings",
    model="all-MiniLM-L6-v2",
    batchSize=32,
)

result = embeddings.transform(text_df)
```

### Embedding Text Chunks

```python
from scaledp.models.splitters.TextSplitter import TextSplitter
from scaledp.models.embeddings.TextEmbeddings import TextEmbeddings
from pyspark.ml import Pipeline

# Create a pipeline: split text then generate embeddings
pipeline = Pipeline(stages=[
    TextSplitter(inputCol="document", outputCol="chunks", chunk_size=500),
    TextEmbeddings(inputCol="chunks", outputCol="embeddings", model="all-MiniLM-L6-v2"),
])

result = pipeline.fit(documents_df).transform(documents_df)
```

## Pipeline Integration

```
Document Input
    ↓
Text Splitter (optional)
    ↓
TextChunks or Raw Text
    ↓
TextEmbeddings Transformer
    ↓
EmbeddingsOutput
    ↓
Downstream Applications (Search, Classification, Clustering, etc.)
```

## Output Format

All embedding transformers produce output in the `EmbeddingsOutput` schema containing:

- **path**: Source document path
- **data**: Embedding vector (list of floats)
- **type**: "text" (raw) or "text_chunk" (from chunks)
- **exception**: Error messages (if any)
- **processing_time**: Computation duration

For detailed schema information, see [EmbeddingsOutput Schema Documentation](schemas/embeddings_output.md).
