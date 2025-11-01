(FaceDetector)=
# FaceDetector

## Overview

`FaceDetector` is a face detection transformer based on the YOLO ONNX model. It is designed to efficiently detect faces in images using a pre-trained model from Hugging Face Hub. The detector is implemented as a PySpark ML transformer and can be integrated into Spark pipelines for scalable face detection tasks.

## Usage Example

```python
from scaledp import FaceDetector, ImageDrawBoxes, PipelineModel

detector = FaceDetector(
        keepInputData=True,
        partitionMap=True,
        numPartitions=0,
        scoreThreshold=0.25,
        task="detect",
        padding=20,
    )

draw = ImageDrawBoxes(
    keepInputData=True,
    inputCols=["image", "boxes"],
    filled=False,
    color="green",
    lineWidth=5,
    displayDataList=[],
)
# Transform the image dataframe through the OCR stage
pipeline = PipelineModel(stages=[detector, draw])
result = pipeline.transform(image_df)
result.show_image("image_with_boxes")
```

![ShowFaceBoxes.png](../../_static/ShowFaceBoxes.png)

## Parameters

| Parameter         | Type    | Description                                      | Default                     |
|-------------------|---------|--------------------------------------------------|-----------------------------|
| inputCol          | str     | Input image column                               | "image"                    |
| outputCol         | str     | Output column for boxes                          | "boxes"                    |
| keepInputData     | bool    | Keep input data in output                        | False                       |
| scaleFactor       | float   | Image resize factor                              | 1.0                         |
| scoreThreshold    | float   | Minimum confidence score                         | 0.2                         |
| device            | Device  | Inference device (CPU/GPU)                       | Device.CPU                  |
| batchSize         | int     | Batch size for inference                         | 2                           |
| partitionMap      | bool    | Use partitioned mapping                          | False                       |
| numPartitions     | int     | Number of partitions                             | 0                           |
| pageCol           | str     | Page column                                      | "page"                     |
| pathCol           | str     | Path column                                      | "path"                     |
| propagateError    | bool    | Propagate errors                                 | False                       |
| task              | str     | Detection task type                              | "detect"                   |
| onlyRotated       | bool    | Return only rotated boxes                        | False                       |
| model             | str     | Model identifier                                 | "StabRise/face_detection"  |

## Notes
- The detector uses the YOLO ONNX model from Hugging Face Hub for face detection.
- Supports batch processing and distributed inference with Spark.
- Additional parameters can be set using the corresponding setter methods.

