(ImageDrawBoxes)=
# ImageDrawBoxes

## Overview

`ImageDrawBoxes` is a PySpark ML transformer that draws bounding boxes and/or NER entity boxes on images. It supports both standard bounding boxes and named entity recognition (NER) outputs, allowing for flexible visualization of detected objects or entities. The transformer can be integrated into Spark pipelines for scalable image annotation tasks.

## Usage Example

```python
from scaledp import FaceDetector, ImageDrawBoxes, PipelineModel

detector = FaceDetector(
    inputCol="image",
    outputCol="boxes",
    keepInputData=True,
    scoreThreshold=0.25,
    padding=20,
)

draw = ImageDrawBoxes(
    inputCols=["image", "boxes"],
    outputCol="image_with_boxes",
    keepInputData=True,
    filled=False,
    color="green",
    lineWidth=5,
)

pipeline = PipelineModel(stages=[detector, draw])
result = pipeline.transform(image_df)
result.show_image("image_with_boxes")
```
![ShowFaceBoxes.png](../_static/ShowFaceBoxes.png)

## Parameters

| Parameter         | Type    | Description                                      | Default             |
|-------------------|---------|--------------------------------------------------|---------------------|
| inputCols         | list    | Input columns: image and boxes/entities           | ["image", "boxes"] |
| outputCol         | str     | Output column for annotated images                | "image_with_boxes" |
| keepInputData     | bool    | Keep input data in output                        | False               |
| imageType         | Enum    | Type of image (e.g., FILE)                       | ImageType.FILE      |
| filled            | bool    | Fill rectangles                                  | False               |
| color             | str     | Box color (hex or name)                          | None (random)       |
| lineWidth         | int     | Line width for boxes                             | 1                   |
| textSize          | int     | Text size for labels                             | 12                  |
| displayDataList   | list    | List of box/entity attributes to display as text | []                  |
| numPartitions     | int     | Number of partitions for Spark                   | 0                   |
| padding           | int     | Padding added to boxes                           | 0                   |
| pageCol           | str     | Page column for repartitioning                   | "page"              |
| whiteList         | list    | Only draw boxes/entities of these types           | []                  |
| blackList         | list    | Do not draw boxes/entities of these types         | []                  |

## Notes
- Supports drawing both standard bounding boxes and NER entity boxes.
- Colors can be set manually or randomly assigned per entity/class.
- Text labels can be displayed using `displayDataList`.
- Handles rotated boxes and fills/outline options.
- Can be used in Spark pipelines for distributed image annotation.
- Errors are handled gracefully and logged.
