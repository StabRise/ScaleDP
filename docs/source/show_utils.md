(ShowUtils)=
# Show Utils

## Overview

Show Utils provides visualization helpers for displaying images, text, PDFs, and named entities from Spark DataFrames in Jupyter/IPython environments. It is designed to work with ScaledP's data structures and transformers, making it easy to inspect and debug results in interactive notebooks.

## Functions

### show_image
Displays images from a DataFrame column. Automatically converts binary columns to images if needed.
- **Parameters:**
  - `df`: Spark DataFrame
  - `column`: Column name (default: auto-detect)
  - `limit`: Number of images to show (default: 5)
  - `width`: Display width in pixels (default: 600)
  - `show_meta`: Show image metadata (default: True)

![ShowImageInvoice.png](_static/ShowImageInvoice.png)

### show_text
Displays text from a DataFrame column, with optional metadata and formatting.
- **Parameters:**
  - `df`: Spark DataFrame
  - `column`: Column name (default: auto-detect)
  - `field`: Field in the text struct (default: "text")
  - `limit`: Number of texts to show (default: 5)
  - `width`: Display width in pixels (default: 800)

### show_json
Displays JSON data from a DataFrame column, pretty-printed and syntax-highlighted.
- **Parameters:**
  - `df`: Spark DataFrame
  - `column`: Column name (default: auto-detect)
  - `field`: Field in the struct (default: "json_data")
  - `limit`: Number of items to show (default: 5)
  - `width`: Display width in pixels (default: 800)

### show_pdf
Displays PDF pages as images from a DataFrame column. Converts binary PDF data to images for visualization.
- **Parameters:**
  - `df`: Spark DataFrame
  - `column`: Column name (default: auto-detect)
  - `limit`: Number of pages to show (default: 5)
  - `width`: Display width in pixels (default: 600)
  - `show_meta`: Show image metadata (default: True)

### show_ner
Displays named entities from a DataFrame column in tabular format.
- **Parameters:**
  - `df`: Spark DataFrame
  - `column`: NER column (default: "ner")
  - `limit`: Number of entities to show (default: 20)
  - `truncate`: Truncate long text (default: False)

### visualize_ner
Visualizes named entities inline within the original text, color-coded by entity type.
- **Parameters:**
  - `df`: Spark DataFrame
  - `column`: NER column (default: "ner")
  - `text_column`: Text column (default: "text")
  - `limit`: Number of rows to show (default: 20)
  - `width`: Display width in pixels (optional)

## Usage Example

```python
from scaledp import *

# Show images from a DataFrame
image_df.show_image(column="image", limit=3)

# Show recognized text
text_df.show_text(column="document", field="text", limit=2)

# Show PDF pages as images
pdf_df.show_pdf(column="content", limit=2)

# Show named entities
ner_df.show_ner(column="ner", limit=10)

# Visualize NER results inline
ner_df.visualize_ner(column="ner", text_column="text", limit=1)
```


## Notes
- Designed for use in Jupyter/IPython environments; uses HTML and Jinja2 templates for rich output.
- Automatically detects column types and applies necessary conversions (e.g., binary to image).
- Handles errors gracefully and displays exceptions in metadata.
- Useful for debugging and inspecting results in interactive data science workflows.
- Available as methods on Spark DataFrames when ScaledP is installed.

