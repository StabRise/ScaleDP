(CraftTextDetector)=
# CraftTextDetector

## Overview

`CraftTextDetector` is a PySpark ML transformer for text detection in images using the CRAFT model. It supports distributed processing in Spark pipelines, batch inference, and optional refiner network postprocessing for improved accuracy. The detector outputs bounding boxes for detected text regions, with options for rotated boxes and threshold tuning.

## Usage Example

```python
from scaledp.models.detectors import CraftTextDetector
from scaledp import TesseractRecognizer, ImageDrawBoxes, PipelineModel

detector = CraftTextDetector(
    device="cpu",
    keepInputData=True,
    partitionMap=True,
    numPartitions=1,
    width=1600,
    scoreThreshold=0.7,
    textThreshold=0.4,
    linkThreshold=0.4,
    withRefiner=True,
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

| Parameter         | Type    | Description                                      | Default         |
|-------------------|---------|--------------------------------------------------|-----------------|
| inputCol          | str     | Input image column                               | "image"        |
| outputCol         | str     | Output column for boxes                          | "boxes"        |
| keepInputData     | bool    | Keep input data in output                        | False           |
| scaleFactor       | float   | Image resize factor                              | 1.0             |
| scoreThreshold    | float   | Minimum confidence score                         | 0.7             |
| textThreshold     | float   | Threshold for text region score                  | 0.4             |
| linkThreshold     | float   | Threshold for link affinity score                | 0.4             |
| sizeThreshold     | int     | Minimum height for detected regions              | -1              |
| width             | int     | Width for image resizing                         | 1280            |
| withRefiner       | bool    | Enable refiner network postprocessing            | False           |
| device            | Device  | Inference device (CPU/GPU)                       | Device.CPU      |
| batchSize         | int     | Batch size for inference                         | 2               |
| partitionMap      | bool    | Use partitioned mapping                          | False           |
| numPartitions     | int     | Number of partitions                             | 0               |
| pageCol           | str     | Page column                                      | "page"         |
| pathCol           | str     | Path column                                      | "path"         |
| propagateError    | bool    | Propagate errors                                 | False           |
| onlyRotated       | bool    | Return only rotated boxes                        | False           |

## Notes
- Supports optional refiner network for improved text box accuracy (`withRefiner`).
- Outputs bounding boxes for detected text regions, including rotated boxes if `onlyRotated` is True.
- Thresholds (`scoreThreshold`, `textThreshold`, `linkThreshold`) can be tuned for different document types.
- Can be integrated with OCR and visualization stages in Spark pipelines.
- Supports batch and distributed processing for scalable text detection.
- Errors are handled gracefully and can be propagated if desired.

