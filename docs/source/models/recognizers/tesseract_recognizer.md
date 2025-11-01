(TesseractRecognizer)=
# TesseractRecognizer

## Overview

`TesseractRecognizer` is a PySpark ML transformer that runs Tesseract OCR on images. It supports multiple languages, Tesseract libraries (`tesserocr` and `pytesseract`), and advanced options such as line orientation detection, formatting, and rotated box handling. The transformer can be integrated into Spark pipelines for scalable and distributed text recognition tasks.

## Usage Example

```python
from scaledp import DocTRTextDetector, TesseractRecognizer, PipelineModel

detector = DocTRTextDetector(
    device="cpu",
    keepInputData=True,
    scoreThreshold=0.1,
    partitionMap=True,
    numPartitions=1,
)

ocr = TesseractRecognizer(
    keepFormatting=True,
    tessLib="tesserocr",  # or "pytesseract"
    lang=["ukr", "eng"],
    scoreThreshold=0.2,
    partitionMap=True,
    numPartitions=1,
    tessDataPath="/usr/share/tesseract-ocr/5/tessdata/",
    onlyRotated=True,
)

pipeline = PipelineModel(stages=[detector, ocr])
result = pipeline.transform(image_df)
for row in result.collect():
    print(row.text.text)  # Recognized text
```

## Parameters

| Parameter           | Type    | Description                                      | Default                                 |
|---------------------|---------|--------------------------------------------------|-----------------------------------------|
| inputCols           | list    | Input columns: image and boxes                   | ["image", "boxes"]                     |
| outputCol           | str     | Output column for recognized text                | "text"                                 |
| keepInputData       | bool    | Keep input data in output                        | False                                   |
| scaleFactor         | float   | Image resize factor                              | 1.0                                     |
| scoreThreshold      | float   | Minimum confidence score                         | 0.5                                     |
| oem                 | int     | OCR engine mode (see Tesseract OEM)              | OEM.DEFAULT                             |
| lang                | list    | List of languages for OCR                        | ["eng"]                                |
| lineTolerance       | int     | Tolerance for line grouping                      | 0                                       |
| keepFormatting      | bool    | Preserve text formatting                         | False                                   |
| tessDataPath        | str     | Path to Tesseract data folder                    | "/usr/share/tesseract-ocr/5/tessdata/"  |
| tessLib             | int/str | Tesseract library to use (TESSEROCR/PYTESSERACT) | TessLib.PYTESSERACT                     |
| partitionMap        | bool    | Use partitioned mapping                          | False                                   |
| numPartitions       | int     | Number of partitions for Spark                   | 0                                       |
| pageCol             | str     | Page column for repartitioning                   | "page"                                  |
| pathCol             | str     | Path column for image metadata                   | "path"                                  |
| detectLineOrientation| bool   | Detect and auto-orient text lines                | True                                    |
| onlyRotated         | bool    | Only return rotated boxes                        | True                                    |
| oriModel            | str     | Model for line orientation detection             | "StabRise/line_orientation_detection_v0.1" |

## Notes
- Supports both `tesserocr` and `pytesseract` libraries for OCR.
- Can process multiple languages and preserve formatting if desired.
- Handles rotated boxes and auto-orients text lines for improved accuracy.
- Errors are handled gracefully and logged; exceptions are included in the output if any occur.
- Can be used in Spark pipelines for distributed OCR processing.

