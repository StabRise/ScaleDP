(DBNetOnnxDetector)=
# DBNetOnnxDetector

## Overview

`DBNetOnnxDetector` is a PySpark ML transformer for text detection in images using the DBNet ONNX model. It supports distributed processing in Spark pipelines and can automatically download models from Hugging Face Hub. The detector outputs bounding boxes for detected text regions, with options for rotated boxes and merging overlapping results.

## Usage Example

```python
from scaledp.models.detectors import DBNetOnnxDetector
from scaledp import TesseractRecognizer, ImageDrawBoxes, PipelineModel

detector = DBNetOnnxDetector(
    model="StabRise/text_detection_dbnet_ml_v0.2",  # Hugging Face model repo
    keepInputData=True,
    onlyRotated=False,
    scoreThreshold=0.2,
)

ocr = TesseractRecognizer(
    inputCols=["image", "boxes"],
    keepFormatting=False,
    keepInputData=True,
    lang=["eng", "spa"],
    scoreThreshold=0.2,
    scaleFactor=2.0,
    partitionMap=True,
    numPartitions=1,
)

draw = ImageDrawBoxes(
    keepInputData=True,
    inputCols=["image", "text"],
    filled=False,
    color="green",
    lineWidth=5,
    displayDataList=["score", "text", "angle"],
)

pipeline = PipelineModel(stages=[detector, ocr, draw])
result = pipeline.transform(image_df)
result.show_image("image_with_boxes")
```

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
| onlyRotated       | bool    | Return only rotated boxes                        | False                       |
| model             | str     | Model identifier or path                         | "StabRise/text_detection_dbnet_ml_v0.2" |

## Notes
- Automatically downloads the ONNX model from Hugging Face Hub if not present locally.
- Outputs bounding boxes for detected text regions, including rotated boxes if `onlyRotated` is True.
- Merges overlapping boxes based on IOU, angle, and line proximity for cleaner results.
- Can be integrated with OCR and visualization stages in Spark pipelines.
- Supports batch and distributed processing for scalable text detection.
- Errors are handled gracefully and can be propagated if desired.

