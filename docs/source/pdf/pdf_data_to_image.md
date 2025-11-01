(PdfDataToImage)=
# PdfDataToImage

## Overview

`PdfDataToImage` is a PySpark ML transformer that extracts images from PDF files, converting each page into an image. It supports both Spark and Pandas DataFrames, configurable resolution, page limits, and output image types. This transformer is useful for document digitization, OCR preprocessing, and distributed PDF-to-image conversion workflows.

## Usage Example

```python
from scaledp.pdf import PdfDataToImage
from pyspark.ml import PipelineModel

pdf_to_image = PdfDataToImage(
    inputCol="content",      # Column with PDF binary data
    outputCol="image",       # Output column for images
    pathCol="path",          # Optional: column with PDF file paths
    pageCol="page",          # Output page number column
    keepInputData=True,       # Keep original data in output
    imageType="FILE",        # Output image type (e.g., FILE, PIL)
    resolution=300,           # DPI for image extraction
    pageLimit=5,              # Limit number of pages processed
)

pipeline = PipelineModel(stages=[pdf_to_image])
result = pipeline.transform(pdf_df)  # pdf_df should have 'content' and optionally 'path' columns
result.show_image("image")
```

## Parameters

| Parameter    | Type | Description                          | Default                |
|--------------|------|--------------------------------------|------------------------|
| inputCol     | str  | Input column with PDF binary data     | "content"             |
| outputCol    | str  | Output column for images              | "image"               |
| pathCol      | str  | Path column for PDF metadata          | "path"                |
| pageCol      | str  | Output page number column             | "page"                |
| keepInputData| bool | Keep input data in output             | False                  |
| imageType    | Enum | Output image type (e.g., FILE, PIL)   | ImageType.FILE         |
| resolution   | int  | DPI for image extraction              | 300                    |
| pageLimit    | int  | Limit number of pages processed       | 0 (no limit)           |

## Notes
- Converts each PDF page to an image using the specified resolution and image type.
- Supports limiting the number of pages processed with `pageLimit`.
- Handles errors gracefully; if an exception occurs, an empty image with the error message is returned.
- Can be used as the first stage in document processing pipelines for OCR or image analysis.
- Supports distributed processing with Spark and Pandas DataFrames.

