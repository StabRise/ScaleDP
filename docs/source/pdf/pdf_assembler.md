(PdfAssembler)=
# PdfAssembler

## Overview

`PdfAssembler` is a PySpark ML transformer that assembles single-page PDF documents into a single multi-page PDF. It supports both Spark and Pandas DataFrames, grouping pages by a specified column (e.g., file path) and merging them using PyMuPDF (fitz). This transformer is useful for reconstructing full documents from page-level PDF outputs in distributed pipelines.

## Usage Example

```python
from scaledp.pdf import PdfAssembler
from pyspark.ml import PipelineModel

pdf_assembler = PdfAssembler(
    inputCol="pdf_with_text_layer",  # Column with single-page PDFs
    outputCol="assembled_pdf",       # Output column for merged PDF
    groupByCol="path",               # Group pages by file path
)

pipeline = PipelineModel(stages=[pdf_assembler])
result = pipeline.transform(pdf_df)
for row in result.collect():
    with open("output.pdf", "wb") as f:
        f.write(row.assembled_pdf.data)
```

## Parameters

| Parameter    | Type | Description                          | Default                |
|--------------|------|--------------------------------------|------------------------|
| inputCol     | str  | Input column with single-page PDFs    | "pdf"                 |
| outputCol    | str  | Output column for assembled PDF       | "assembled_pdf"       |
| groupByCol   | str  | Column to group pages by              | "path"                |

## Notes
- Supports both Spark and Pandas DataFrames for flexible integration.
- Groups single-page PDFs by the specified column and merges them in order.
- Uses PyMuPDF (fitz) for PDF manipulation and merging.
- Handles errors gracefully; exceptions are included in the output if any occur.
- Can be used as the final stage in PDF processing pipelines to reconstruct full documents.

## Complex Pipeline Example

This example demonstrates a full pipeline for processing PDFs: converting pages to images, running OCR, adding a text layer, and assembling the final document.

```python
from scaledp.pdf import PdfDataToImage, PdfAddTextLayer, PdfAssembler, SingleImageToPdf
from scaledp import TesseractOcr
from pyspark.ml import PipelineModel

# Step 1: Convert PDF pages to images
pdf_data_to_image = PdfDataToImage(
    inputCol="content",
    outputCol="image",
    pageLimit=10,  # Limit number of pages processed
)

# Step 2: Run OCR on images
ocr = TesseractOcr(
    inputCol="image",
    outputCol="text",
    keepInputData=True,
    tessLib="tesserocr",  # or "pytesseract"
    lang=["eng", "spa"],
    scoreThreshold=0.2,
)

# Step 3: Convert images back to single-page PDFs
image_to_pdf = SingleImageToPdf(
    inputCol="image",
    outputCol="pdf",
)

# Step 4: Add recognized text as a layer to each PDF page
pdf_text_layer = PdfAddTextLayer(
    inputCols=["pdf", "text"],
    outputCol="pdf_with_text_layer",
)

# Step 5: Assemble all processed pages into a single PDF document
pdf_assembler = PdfAssembler(
    inputCol="pdf_with_text_layer",
    outputCol="assembled_pdf",
    groupByCol="path",
)

# Build and run the pipeline
pipeline = PipelineModel(stages=[
    pdf_data_to_image,
    ocr,
    image_to_pdf,
    pdf_text_layer,
    pdf_assembler,
])
result = pipeline.transform(pdf_df)

# Save the assembled PDF
for row in result.collect():
    with open("output.pdf", "wb") as f:
        f.write(row.assembled_pdf.data)
```

This pipeline:
- Converts PDF pages to images
- Runs OCR to extract text
- Adds the text layer to each PDF page
- Assembles all processed pages into a single PDF document
- Works with both Spark and Pandas DataFrames
