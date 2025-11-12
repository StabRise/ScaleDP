(BaseEmbeddings)=
# BaseEmbeddings

## Overview

`BaseEmbeddings` is an abstract base class for embedding transformers in ScaleDP. It provides the foundational structure and common functionality for embedding models, enabling efficient and scalable embedding generation for various data types. Derived classes, such as `TextEmbeddings`, extend this base class to implement specific embedding logic.

## Key Features

- **Abstract Base Class**: Provides a common interface for embedding transformers.
- **PySpark Integration**: Designed to work seamlessly with PySpark for distributed data processing.
- **Customizable Parameters**: Supports a wide range of parameters for flexibility and customization.
- **Error Handling**: Includes validation for input columns and error propagation options.

## Usage Example

`BaseEmbeddings` is not intended to be used directly. Instead, it serves as a parent class for specific embedding transformers like `TextEmbeddings`.

## Parameters

| Parameter         | Type    | Description                                      | Default                     |
|-------------------|---------|--------------------------------------------------|-----------------------------|
| inputCol          | str     | Input column name                                | N/A                         |
| outputCol         | str     | Output column name                               | N/A                         |
| keepInputData     | bool    | Whether to retain input data in the output       | True                        |
| device            | Device  | Device for computation (CPU/GPU)                | Device.CPU                  |
| model             | str     | Pre-trained model identifier                     | N/A                         |
| batchSize         | int     | Batch size for processing                        | 1                           |
| numPartitions     | int     | Number of partitions for distributed processing  | 1                           |
| partitionMap      | bool    | Use partitioned mapping                          | False                       |
| pageCol           | str     | Page column                                      | "page"                    |
| pathCol           | str     | Path column                                      | "path"                    |

## Notes

- `BaseEmbeddings` provides the `_transform` method, which handles the core logic for applying transformations to a dataset.
- Derived classes must implement the `transform_udf` and `transform_udf_pandas` methods to define the specific embedding logic.
- The class includes validation for input columns to ensure compatibility with the dataset.

