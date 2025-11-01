(ImageCropBoxes)=
# ImageCropBoxes

## Overview

`ImageCropBoxes` is a PySpark ML transformer that crops images based on provided bounding boxes. It is designed to process images in Spark pipelines, supporting batch and distributed processing. The transformer can add padding to crops, limit the number of crops per image, and handle cases where no boxes are present.

## Usage Example

```python
from scaledp import FaceDetector, ImageCropBoxes, PipelineModel

# Step 1: Detect faces in images
detector = FaceDetector(
    inputCol="image",
    outputCol="boxes",
    keepInputData=True,
    scoreThreshold=0.25,
    padding=20,
)

# Step 2: Crop images using detected face boxes
cropper = ImageCropBoxes(
    inputCols=["image", "boxes"],
    outputCol="cropped_image",
    keepInputData=True,
    padding=10,
    limit=5,
    noCrop=True,
    autoRotate=False,  # Automatically rotate crops if box height > width
)

# Build and run the pipeline
pipeline = PipelineModel(stages=[detector, cropper])
result = pipeline.transform(image_df)
result.show_image("cropped_image")
```

![ShowFaceCropped.png](../_static/ShowFaceCropped.png)

## Parameters

| Parameter         | Type    | Description                                      | Default         |
|-------------------|---------|--------------------------------------------------|-----------------|
| inputCols         | list    | Input columns: image and boxes                   | ["image", "boxes"] |
| outputCol         | str     | Output column for cropped images                 | "cropped_image"|
| keepInputData     | bool    | Keep input data in output                        | False           |
| imageType         | Enum    | Type of image (e.g., FILE)                       | ImageType.FILE  |
| numPartitions     | int     | Number of partitions for Spark                   | 0               |
| padding           | int     | Padding added to each crop                       | 0               |
| pageCol           | str     | Page column for repartitioning                   | "page"          |
| propagateError    | bool    | Propagate errors                                 | False           |
| noCrop            | bool    | Raise error if no boxes to crop                  | True            |
| limit             | int     | Limit number of crops per image                  | 0 (no limit)    |
| autoRotate        | bool    | Auto rotate crop if box height > width           | True            |

## Notes
- Crops are performed using bounding boxes from the `boxes` column.
- If `noCrop` is True and no boxes are present, an error is raised.
- If `limit` is set, only the first N boxes are used for cropping.
- If `autoRotate` is True, crops are rotated if the bounding box height is greater than its width.
- Supports distributed processing with Spark.
- Errors can be propagated or handled gracefully based on `propagateError`.
