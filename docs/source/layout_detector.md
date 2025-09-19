# Layout Detection

The LayoutDetector is a powerful component in ScaleDP that uses PaddleOCR's layout analysis capabilities to detect and classify different regions within documents. This detector can identify various layout types such as text blocks, titles, lists, tables, and figures.

## Overview

Layout detection is essential for understanding document structure and extracting meaningful information from complex documents. The LayoutDetector provides:

- **Multiple Layout Types**: Detects text, titles, lists, tables, and figures
- **Configurable Detection**: Customize which types to detect and confidence thresholds
- **GPU Acceleration**: Support for GPU processing to improve performance
- **Integration**: Seamless integration with ScaleDP pipeline
- **Error Handling**: Robust error handling for various edge cases

## Installation

The LayoutDetector requires PaddleOCR to be installed:

```bash
pip install paddleocr
```

## Basic Usage

### Initialize the LayoutDetector

```python
from scaledp.models.detectors.LayoutDetector import LayoutDetector
from scaledp.enums import Device

# Create a LayoutDetector instance
layout_detector = LayoutDetector(
    inputCol="image",
    outputCol="layout_boxes",
    scoreThreshold=0.5,  # Confidence threshold
    device=Device.CPU,   # Use CPU for inference
    whiteList=["text", "title", "list", "table", "figure"],  # Types to detect
    model="PP-DocLayout_plus-L"  # Model to use
)
```

### Process an Image

```python
from scaledp.schemas.Image import Image
from PIL import Image as PILImage

# Load and prepare image
pil_image = PILImage.open("document.png")
image = Image(
    path="document.png",
    data=pil_image,
    exception=""
)

# Run layout detection
result = layout_detector.transform_udf(image)

# Access results
print(f"Detected {len(result.bboxes)} layout regions")
for box in result.bboxes:
    print(f"- {box.text}: confidence {box.score:.3f}")
```

## Configuration Parameters

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `inputCol` | str | "image" | Input column name containing images |
| `outputCol` | str | "layout_boxes" | Output column name for detection results |
| `scoreThreshold` | float | 0.5 | Minimum confidence score for detections |
| `device` | Device | Device.CPU | Processing device (CPU/GPU) |
| `whiteList` | List[str] | ["text", "title", "list", "table", "figure"] | Layout types to detect |
| `model` | str | "PP-DocLayout_plus-L" | PaddleOCR layout detection model name |

### Advanced Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `scaleFactor` | float | 1.0 | Image scaling factor |
| `keepInputData` | bool | False | Whether to keep input data in output |
| `partitionMap` | bool | False | Enable partitioned processing |
| `numPartitions` | int | 0 | Number of partitions for processing |
| `propagateError` | bool | False | Whether to propagate errors |

## Available Models

The LayoutDetector supports different PaddleOCR layout detection models:

- **PP-DocLayout_plus-L**: Large model with high accuracy (default)
- **PP-DocLayout-M**: Medium model with balanced speed and accuracy

## Layout Types

The LayoutDetector can identify the following layout types:

- **text**: General text content
- **title**: Document titles and headings
- **list**: Bulleted or numbered lists
- **table**: Tabular data structures
- **figure**: Images, charts, and diagrams

## Examples

### Custom Layout Type Detection

```python
# Detect only text and tables
text_table_detector = LayoutDetector(
    inputCol="image",
    outputCol="text_table_boxes",
    scoreThreshold=0.6,
    whiteList=["text", "table"],
    model="PP-DocLayout-M"  # Use medium model for faster processing
)
```

### GPU Acceleration

```python
# Use GPU for faster processing
gpu_detector = LayoutDetector(
    inputCol="image",
    outputCol="gpu_layout_boxes",
    device=Device.CUDA,
    scoreThreshold=0.5
)
```

### Pipeline Integration

```python
from pyspark.ml import PipelineModel
from scaledp.models.image.DataToImage import DataToImage

pipeline = PipelineModel(stages=[
    DataToImage(inputCol="content", outputCol="image"),
    LayoutDetector(
        inputCol="image",
        outputCol="layout_boxes",
        scoreThreshold=0.5
    )
])

result = pipeline.transform(df)
```

## Output Format

The LayoutDetector returns a `DetectorOutput` object containing:

- **path**: Image file path
- **type**: Detection type ("layout")
- **bboxes**: List of detected layout regions
- **exception**: Any error messages

Each detected region includes:

- **text**: Layout type (text, title, list, table, figure)
- **score**: Confidence score (0.0 to 1.0)
- **x, y**: Top-left coordinates
- **width, height**: Region dimensions
- **polygon**: Optional polygon coordinates for rotated regions

## Performance Considerations

### CPU vs GPU

- **CPU**: Suitable for small batches and development
- **GPU**: Recommended for production and large-scale processing

### Batch Processing

For large datasets, consider using partitioned processing:

```python
layout_detector = LayoutDetector(
    inputCol="image",
    outputCol="layout_boxes",
    partitionMap=True,
    numPartitions=4
)
```

### Memory Management

The detector automatically handles memory cleanup, but for very large images, consider:

- Using `scaleFactor` to reduce image size
- Processing in smaller batches
- Monitoring memory usage

## Error Handling

The LayoutDetector includes robust error handling:

- **Import Errors**: Graceful handling when PaddleOCR is not installed
- **Processing Errors**: Individual image errors don't stop batch processing
- **Configuration Errors**: Clear error messages for invalid parameters

## Use Cases

### Document Analysis

```python
# Analyze document structure
result = layout_detector.transform_udf(document_image)

# Extract titles
titles = [box for box in result.bboxes if box.text == "title"]

# Extract tables
tables = [box for box in result.bboxes if box.text == "table"]
```

### Content Extraction

```python
# Focus on specific content types
text_detector = LayoutDetector(
    inputCol="image",
    outputCol="text_regions",
    layoutTypes=["text", "title"]
)
```

### Quality Control

```python
# High confidence detection
high_confidence_detector = LayoutDetector(
    inputCol="image",
    outputCol="high_conf_boxes",
    scoreThreshold=0.8
)
```

## Troubleshooting

### Common Issues

1. **PaddleOCR not installed**
   ```
   pip install paddleocr
   ```

2. **GPU not available**
   - Check CUDA installation
   - Verify PaddleOCR GPU support
   - Fall back to CPU processing

3. **Memory issues**
   - Reduce `scaleFactor`
   - Process smaller batches
   - Monitor system resources

### Performance Tips

- Use GPU when available for faster processing
- Adjust `scoreThreshold` based on quality requirements
- Consider image preprocessing for better results
- Use appropriate batch sizes for your hardware

## Integration with Other Components

The LayoutDetector works well with other ScaleDP components:

- **OCR**: Extract text from detected text regions
- **NER**: Apply named entity recognition to text regions
- **Visual Extractors**: Extract data from specific layout types
- **Image Processing**: Draw bounding boxes around detected regions
