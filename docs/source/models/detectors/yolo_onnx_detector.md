(YoloOnnxDetector)=
# YoloOnnxDetector

## Overview

`YoloOnnxDetector` is a generic object detector transformer based on the YOLO ONNX model. It provides efficient detection of objects in images using a pre-trained YOLO model, supporting batch and distributed inference in Spark pipelines. It is designed for extensibility and is used as a base for specialized detectors such as FaceDetector and SignatureDetector.
It not need installing Pytorch, only ONNX Runtime is required.


## Inheritance

- Inherits from [`BaseDetector`](./BaseDetector.md), which provides core Spark ML transformer functionality and schema handling.
- Mixes in `HasDevice` and `HasBatchSize` for device and batch configuration.

## Usage Example

```python
from scaledp.models.detectors.YoloOnnxDetector import YoloOnnxDetector

detector = YoloOnnxDetector(
    model="StabRise/face_detection",  # or any supported YOLO ONNX model
    scoreThreshold=0.3,
    padding=10,
)

# Use in a Spark pipeline
detected_df = detector.transform(input_df)
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
| task              | str     | Detection task type                              | "detect"                   |
| onlyRotated       | bool    | Return only rotated boxes                        | False                       |
| model             | str     | Model identifier                                 | (required)                  |
| padding           | int     | Padding percent to expand detected boxes         | 0                           |

## Notes
- The detector loads YOLO ONNX models from Hugging Face Hub or local path.
- Supports batch and distributed processing with Spark.
- Padding expands detected bounding boxes by a percentage.
- Used as a base for specialized detectors (e.g., [**Face Detector**]
  (#FaceDetector), [**Signature Detector**](#SignatureDetector)).

