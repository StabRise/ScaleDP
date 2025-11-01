(PdfDataToText)=
# PdfDataToText

## Overview

`PdfDataToText` is a PySpark ML transformer that extracts text and word-level bounding boxes from PDF files. It processes each page of a PDF, returning both the text content and the coordinates of each word, making it suitable for downstream tasks such as OCR, document analysis, and layout understanding. The transformer supports both Spark and Pandas DataFrames and handles errors gracefully.

## Usage Example

```python
from scaledp.pdf import PdfDataToText
from pyspark.ml import PipelineModel

pdf_to_text = PdfDataToText(
    inputCol="content",      # Column with PDF binary data
    outputCol="document",    # Output column for extracted text and boxes
    pathCol="path",          # Optional: column with PDF file paths
    pageCol="page",          # Output page number column
    keepInputData=True,       # Keep original data in output
)

pipeline = PipelineModel(stages=[pdf_to_text])
result = pipeline.transform(pdf_df)  # pdf_df should have 'content' and optionally 'path' columns
for row in result.collect():
    print(row.document.text)  # Extracted text
    print(row.document.bboxes)  # List of word bounding boxes
```

## Parameters

| Parameter      | Type | Description                          | Default      |
|----------------|------|--------------------------------------|--------------|
| inputCol       | str  | Input column with PDF binary data     | "content"    |
| outputCol      | str  | Output column for extracted text      | "document"   |
| pathCol        | str  | Path column for PDF metadata          | "path"       |
| pageCol        | str  | Output page number column             | "page"       |
| keepInputData  | bool | Keep input data in output             | False        |

## Notes
- Extracts text and word-level bounding boxes for each page in the PDF.
- Returns a `Document` object with `text`, `bboxes`, and metadata for each page.
- Handles errors gracefully; if an exception occurs, an empty document with the error message is returned.
- Can be used as the first stage in document analysis or OCR pipelines.
- Supports distributed processing with Spark and Pandas DataFrames.

