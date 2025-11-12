(TextEmbeddings)=
# TextEmbeddings

## Overview

`TextEmbeddings` is a text embedding transformer based on the SentenceTransformer model. It is designed to efficiently generate embeddings for text data using a pre-trained model. The transformer is implemented as a PySpark ML transformer and can be integrated into Spark pipelines for scalable text embedding tasks.

## Usage Example

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

## Notes
- The transformer uses the SentenceTransformer model for generating text embeddings.
- Supports batch processing and distributed inference with Spark.
- Additional parameters can be set using the corresponding setter methods.
