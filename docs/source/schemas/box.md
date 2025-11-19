(Box)=
# Box Schema

## Overview

`Box` is a structured data schema that represents a bounding box with positional and text information. It is used to store layout information such as text regions, detected objects, or other spatial elements within documents. Boxes are typically part of a Document schema when processing images or PDFs with layout awareness.

## Schema Structure

```python
from scaledp.schemas.Box import Box
from dataclasses import dataclass

# Box dataclass definition
@dataclass(order=True)
class Box:
    text: str              # Text content of the box
    score: float           # Confidence score
    x: int                 # X coordinate (top-left)
    y: int                 # Y coordinate (top-left)
    width: int             # Box width in pixels
    height: int            # Box height in pixels
    angle: float = 0.0     # Rotation angle in degrees (default 0)
```

## Fields

| Field  | Type  | Default | Description                                    |
|--------|-------|---------|------------------------------------------------|
| text   | str   | -       | Text content or label of the box              |
| score  | float | -       | Confidence score (typically 0.0 to 1.0)       |
| x      | int   | -       | X coordinate of top-left corner               |
| y      | int   | -       | Y coordinate of top-left corner               |
| width  | int   | -       | Width of the bounding box                     |
| height | int   | -       | Height of the bounding box                    |
| angle  | float | 0.0     | Rotation angle in degrees (-180 to 180)       |

## Usage Examples

### Creating Boxes

```python
from scaledp.schemas.Box import Box

# Simple axis-aligned box
box = Box(
    text="Title",
    score=0.95,
    x=10,
    y=20,
    width=300,
    height=50,
    angle=0.0
)

# Rotated box
rotated_box = Box(
    text="Rotated Text",
    score=0.89,
    x=100,
    y=150,
    width=200,
    height=30,
    angle=45.0  # 45 degrees rotation
)
```

### Creating from Bounding Box Coordinates

```python
from scaledp.schemas.Box import Box

# From [x1, y1, x2, y2] format
bbox = [10, 20, 110, 70]  # x1, y1, x2, y2

box = Box.from_bbox(
    box=bbox,
    angle=0.0,
    label="Text",
    score=0.92
)
# Result: Box(text="Text", score=0.92, x=10, y=20, width=100, height=50, angle=0.0)
```

### Creating from Polygon Points

```python
from scaledp.schemas.Box import Box

# From polygon coordinates (quadrilateral)
polygon_points = [
    (10, 20),    # top-left
    (310, 20),   # top-right
    (310, 70),   # bottom-right
    (10, 70)     # bottom-left
]

box = Box.from_polygon(
    polygon_points=polygon_points,
    text="Box Text",
    score=0.95,
    padding=0
)
```

## Methods

### Bounding Box Operations

#### `to_string()`
```python
box = Box(text="Title", score=0.95, x=10, y=20, width=100, height=50)
box = box.to_string()
# Ensures text is string type
```

#### `scale(factor, padding)`
```python
# Scale box by factor and add padding
original = Box(text="Text", score=0.95, x=100, y=100, width=200, height=50)

# Scale by 2.0 (double size) and add 10px padding
scaled = original.scale(factor=2.0, padding=10)
# Result: x=190, y=190, width=410, height=110
```

#### `shape(padding)`
```python
box = Box(text="Text", score=0.95, x=10, y=20, width=100, height=50)

# Get corner coordinates
corners = box.shape(padding=0)
# Result: [(10, 20), (110, 70)]

# With padding
corners_padded = box.shape(padding=5)
# Result: [(5, 15), (115, 75)]
```

#### `bbox(padding)`
```python
box = Box(text="Text", score=0.95, x=10, y=20, width=100, height=50)

# Get bbox coordinates [x1, y1, x2, y2]
bbox = box.bbox(padding=0)
# Result: [10, 20, 110, 70]

# With padding
bbox_padded = box.bbox(padding=5)
# Result: [5, 15, 115, 75]
```

### Geometric Operations

#### `is_rotated()`
```python
# Check if box is rotated (angle >= 3 degrees)
box = Box(..., angle=45.0)
if box.is_rotated():
    print("Box is rotated")
```

#### `iou(box1, box2)`
```python
from scaledp.schemas.Box import Box

box1 = Box(text="A", score=0.95, x=0, y=0, width=100, height=100)
box2 = Box(text="B", score=0.92, x=50, y=50, width=100, height=100)

# Calculate Intersection over Union
iou = Box.iou(box1, box2)
# Result: 0.14... (25% overlap)
```

#### `merge(box1, box2)`
```python
from scaledp.schemas.Box import Box

box1 = Box(text="A", score=0.95, x=10, y=10, width=100, height=50)
box2 = Box(text="B", score=0.92, x=80, y=40, width=100, height=50)

# Merge boxes (returns minimal bounding rectangle)
merged = Box.merge(box1, box2)
# Result: Box(text="A", score=0.95, x=10, y=10, width=170, height=80)
```

### Batch Operations

#### `is_on_same_line(box1, box2, angle_thresh, line_thresh)`
```python
from scaledp.schemas.Box import Box

box1 = Box(text="Word1", score=0.95, x=0, y=0, width=50, height=20, angle=0)
box2 = Box(text="Word2", score=0.92, x=60, y=2, width=50, height=20, angle=0)

# Check if boxes are on same text line
same_line = Box.is_on_same_line(
    box1, box2,
    angle_thresh=10.0,    # Max angle difference
    line_thresh=0.5       # Max normalized center difference
)
# Result: True (boxes are roughly aligned)
```

#### `merge_overlapping_boxes(boxes, iou_threshold, angle_thresh, line_thresh)`
```python
from scaledp.schemas.Box import Box

boxes = [
    Box(text="A", score=0.95, x=0, y=0, width=100, height=50),
    Box(text="B", score=0.92, x=30, y=10, width=100, height=50),  # Overlaps with A
    Box(text="C", score=0.90, x=200, y=0, width=100, height=50),  # No overlap
]

# Merge overlapping boxes
merged = Box.merge_overlapping_boxes(
    boxes,
    iou_threshold=0.3,
    angle_thresh=10.0,
    line_thresh=0.5
)
# Result: [merged_A_B, C]
```

## Related Schemas

- [Document](./document.md) - Contains list of boxes
- [TextChunks](./text_chunks.md) - Output schema

## See Also

- [Document Schema](./document.md)
- [Text Splitters](../models/splitters/index.md)
- [Image Processing](../image_processing.md)
