(BaseSplitter)=
# BaseSplitter

## Overview

`BaseSplitter` is the abstract base class for all text splitter transformers in the ScaleDP library. It extends PySpark's `Transformer` class and provides common functionality for splitting documents into chunks. This class defines the interface and shared parameters for all splitter implementations.

## Inheritance

- Extends PySpark's `Transformer` for ML pipeline compatibility.
- Mixes in the following parameter mixins:
  - `HasInputCol` - Input column containing documents
  - `HasOutputCol` - Output column for results
  - `HasKeepInputData` - Whether to preserve input data
  - `HasChunkSize` - Maximum chunk size
  - `HasChunkOverlap` - Overlap between chunks
  - `HasNumPartitions` - Partition control
  - `HasPartitionMap` - Enable distributed processing mode
  - `HasWhiteList` - Whitelist filtering support

## Key Features

- **PySpark Integration**: Full compatibility with PySpark ML pipelines
- **Serialization**: Support for reading and writing model parameters
- **Flexible Configuration**: Extensive parameters for customization
- **Extensible Design**: Foundation for specialized splitter implementations
- **Batch Processing**: Support for both local and distributed processing modes

## Class Hierarchy

```
BaseSplitter
├── BaseTextSplitter
│   └── TextSplitter (concrete implementation)
```

## Parameters

| Parameter         | Type    | Description                                      | Default                     |
|-------------------|---------|--------------------------------------------------|-----------------------------|
| inputCol          | str     | Input column name                                | varies by implementation    |
| outputCol         | str     | Output column name                               | varies by implementation    |
| keepInputData     | bool    | Keep input columns in output                     | True                        |
| chunkSize         | int     | Size of each chunk                               | 500                         |
| chunkOverlap      | int     | Overlap between consecutive chunks               | 0                           |
| numPartitions     | int     | Number of partitions                             | 1                           |
| partitionMap      | bool    | Use partitioned mapping (pandas_udf mode)        | False                       |
| whiteList         | list    | Whitelist of allowed items                       | []                          |

## Abstract Methods

Subclasses must implement the following abstract methods:

### transform(dataset)
Transforms a Spark DataFrame by applying the splitter logic.

**Parameters:**
- `dataset` (pyspark.sql.DataFrame): Input DataFrame

**Returns:**
- (pyspark.sql.DataFrame): DataFrame with split results

## Usage Guidelines

`BaseSplitter` is an abstract class and should not be instantiated directly. Instead, use concrete implementations like:

- [`TextSplitter`](./text_splitter.md) - Semantic text splitting

```python
# Correct: Use concrete implementation
from scaledp.models.splitters.TextSplitter import TextSplitter

splitter = TextSplitter(chunk_size=500, chunk_overlap=50)
```

```python
# Incorrect: Do not instantiate BaseSplitter directly
from scaledp.models.splitters.BaseSplitter import BaseSplitter

# This will raise an error
splitter = BaseSplitter()  # Error!
```

## Creating Custom Splitters

To create a custom splitter, inherit from `BaseSplitter` or `BaseTextSplitter`:

```python
from scaledp.models.splitters.BaseTextSplitter import BaseTextSplitter
from scaledp.schemas.Document import Document
from scaledp.schemas.TextChunks import TextChunks

class CustomSplitter(BaseTextSplitter):
    """Custom splitter implementation."""

    def split(self, document: Document) -> TextChunks:
        """Implement custom splitting logic."""
        # Your splitting algorithm here
        chunks = self._split_text(document.text)
        return TextChunks(
            path=document.path,
            chunks=chunks,
            exception="",
            processing_time=0.0
        )
```

## Pipeline Integration

`BaseSplitter` and its subclasses are designed to work seamlessly with PySpark pipelines:

```python
from pyspark.ml import Pipeline
from scaledp.models.splitters.TextSplitter import TextSplitter

# Create pipeline stages
splitter = TextSplitter(chunk_size=500)

# Create and fit pipeline
pipeline = Pipeline(stages=[splitter])
model = pipeline.fit(training_data)

# Transform data
results = model.transform(test_data)
```

## Serialization

All splitters support PySpark's read/write functionality:

```python
# Save a model
splitter = TextSplitter(chunk_size=500)
splitter.write().overwrite().save("path/to/splitter")

# Load a model
loaded_splitter = TextSplitter.load("path/to/splitter")
```

## Related Classes

- [`BaseTextSplitter`](./base_text_splitter.md) - Abstract base for text splitters
- [`TextSplitter`](./text_splitter.md) - Concrete semantic text splitter implementation
- [`Document`](#Document) - Input document schema
- [`TextChunks`](#TextChunks) - Output text chunks schema

## See Also

- [PySpark ML Transformers](https://spark.apache.org/docs/latest/ml-pipeline.html)
- [Transformer API](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.Transformer.html)
