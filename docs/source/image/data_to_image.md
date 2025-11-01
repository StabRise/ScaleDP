(DataToImage)=
# DataToImage

## Overview

`DataToImage` is a PySpark ML transformer that converts binary content (such as bytes from files or streams) into image objects. It is designed for use in Spark pipelines, enabling scalable and distributed image processing workflows. The transformer supports various image types and handles errors gracefully.

## Usage Example

```python
from scaledp import DataToImage, PipelineModel

image_example = files('resources/images/Invoice.png')

df = spark.read.format("binaryFile") \
    .load(image_example)

data_to_image = DataToImage(
    inputCol="content",      # Column with binary data
    outputCol="image",       # Output column for image objects
    pathCol="path",          # Optional: column with image paths
    keepInputData=True,       # Keep original data in output
    propagateError=False,     # Handle errors gracefully
)

pipeline = PipelineModel(stages=[data_to_image])
result = pipeline.transform(df)  # df should have 'content' and optionally 'path' columns
result.show_image("image")
```

![ShowImageInvoice.png](../_static/ShowImageInvoice.png)

## Parameters

| Parameter         | Type    | Description                                      | Default         |
|-------------------|---------|--------------------------------------------------|-----------------|
| inputCol          | str     | Input column with binary content                  | "content"      |
| outputCol         | str     | Output column for image objects                   | "image"        |
| pathCol           | str     | Path column for image metadata                    | "path"         |
| keepInputData     | bool    | Keep input data in output                        | False           |
| imageType         | Enum    | Type of image (e.g., FILE, PIL)                  | ImageType.FILE  |
| propagateError    | bool    | Propagate errors                                 | False           |

## Notes
- Converts binary data to image objects using the specified image type.
- Handles errors gracefully; if `propagateError` is False, exceptions are logged and empty images are returned.
- Can be used as the first stage in image processing pipelines to ingest raw image data.
- Supports distributed processing with Spark.

