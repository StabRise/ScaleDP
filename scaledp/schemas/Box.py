from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from scaledp.utils.dataclass import map_dataclass_to_struct, register_type


@dataclass(order=True)
class Box:
    """Box object for represent bounding box data in Spark Dataframe."""

    text: str
    score: float
    x: int
    y: int
    width: int
    height: int
    angle: float = 0.0

    def to_string(self) -> "Box":
        self.text = str(self.text)
        return self

    def json(self) -> Dict[str, Any]:
        return {"text": self.text}

    @staticmethod
    def get_schema():
        return map_dataclass_to_struct(Box)

    def scale(self, factor: float, padding: int = 0) -> "Box":
        return Box(
            text=self.text,
            score=self.score,
            x=int(self.x * factor) - padding,
            y=int(self.y * factor) - padding,
            width=int(self.width * factor) + padding,
            height=int(self.height * factor) + padding,
            angle=self.angle,
        )

    def shape(self, padding: int = 0) -> list[tuple[int, int]]:
        return [
            (self.x - padding, self.y - padding),
            (self.x + self.width + padding, self.y + self.height + padding),
        ]

    def bbox(self, padding: int = 0) -> list[int]:
        return [
            self.x - padding,
            self.y - padding,
            self.x + self.width + padding,
            self.y + self.height + padding,
        ]

    @staticmethod
    def from_bbox(box: list[int], angle: float = 0, label: str = "", score: float = 0):
        return Box(
            text=label,
            score=float(score),
            x=int(box[0]),
            y=int(box[1]),
            width=int(box[2] - box[0]),
            height=int(box[3] - box[1]),
            angle=angle,
        )

    @classmethod
    def from_polygon(
        cls,
        polygon_points: list[tuple[float, float]],
        text: str = "",
        score: float = 1.0,
        padding: int = 0,
    ) -> "Box":
        """
        Creates a Box instance from a list of polygon points (typically 4 for a rectangle).
        Uses OpenCV's minAreaRect to find the rotated bounding box properties.

        Args:
            polygon_points (list[tuple[float, float]]): A list of (x, y) coordinates
                                                        representing the vertices of the polygon.
                                                        Expected to be 4 points for a rectangle.
            text (str): Optional text to assign to the box. Defaults to "".
            score (float): Optional score to assign to the box. Defaults to 1.0.

        Returns:
            Box: A new Box instance representing the minimum enclosing rotated rectangle.

        Raises:
            ValueError: If the number of points is not 4.
        """
        import cv2

        if len(polygon_points) != 4:
            # You might allow more points for convex hull, but for a strict 'Box'
            # (which is a rectangle), 4 points are expected.
            raise ValueError(
                "from_polygon expects exactly 4 points for a rectangular box.",
            )

        # Convert list of tuples to a NumPy array for OpenCV
        points_np = np.array(polygon_points, dtype=np.float32)

        # Get the minimum area rotated rectangle
        (center_x, center_y), (raw_width, raw_height), angle_opencv = cv2.minAreaRect(
            points_np,
        )

        # --- Normalize OpenCV's angle and dimensions to our Box convention ---
        # Our convention: width is the horizontal dimension at 0 degrees, angle 0-360 positive
        # CCW.

        box_width, box_height = raw_width, raw_height
        box_angle = angle_opencv

        # If width is smaller than height, swap dimensions and adjust angle by 90 degrees.
        # This makes 'width' conceptually the longer side or the side oriented towards 0/180
        # degrees.
        if box_width < box_height:
            box_width, box_height = box_height, box_width
            box_angle -= 90.0

        # Normalize angle to 0-360 range, ensuring positive counter-clockwise
        # This handles negative angles from OpenCV and converts to our convention
        box_angle = (box_angle % 360 + 360) % 360

        if box_angle > 270:
            box_angle -= 360

        # --- Derive x, y (top-left of the unrotated box) ---
        # The center is center_x, center_y
        # The top-left of the unrotated box is center - (width/2, height/2)
        # So x = center_x - width/2, y = center_y - height/2

        x = int(round(center_x - box_width / 2.0))
        y = int(round(center_y - box_height / 2.0))

        # Ensure dimensions are integers and positive
        box_width_int = int(round(box_width))
        box_height_int = int(round(box_height))
        box_width_int = max(1, box_width_int)
        box_height_int = max(1, box_height_int)

        return cls(
            text=text,
            score=score,
            x=x - padding,
            y=y - padding,
            width=box_width_int + 2 * padding,
            height=box_height_int + 2 * padding,
            angle=box_angle,
        )

    def is_rotated(self) -> bool:
        return abs(self.angle) >= 3

    @staticmethod
    def iou(box1: "Box", box2: "Box") -> float:
        """Compute Intersection over Union (IoU) between two boxes."""
        x1 = max(box1.x, box2.x)
        y1 = max(box1.y, box2.y)
        x2 = min(box1.x + box1.width, box2.x + box2.width)
        y2 = min(box1.y + box1.height, box2.y + box2.height)
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        if inter_area == 0:
            return 0.0
        box1_area = box1.width * box1.height
        box2_area = box2.width * box2.height
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area

    @staticmethod
    def merge(box1: "Box", box2: "Box") -> "Box":
        """Merge two boxes into one by taking the minimal bounding rectangle."""
        x1 = min(box1.x, box2.x)
        y1 = min(box1.y, box2.y)
        x2 = max(box1.x + box1.width, box2.x + box2.width)
        y2 = max(box1.y + box1.height, box2.y + box2.height)
        return Box(
            text=box1.text or box2.text,
            score=max(box1.score, box2.score),
            x=x1,
            y=y1,
            width=x2 - x1,
            height=y2 - y1,
            angle=0.0,
        )

    @staticmethod
    def is_on_same_line(
        box1: "Box",
        box2: "Box",
        angle_thresh: float = 10.0,
        line_thresh: float = 0.5,
    ) -> bool:
        """Check if two boxes are on the same text line.

        - angle_thresh: maximum allowed angle difference (degrees)
        - line_thresh: maximum allowed normalized center difference
        (as a fraction of height for horizontal text)
        """
        # Check angle similarity
        ret = None
        if abs(box1.angle - box2.angle) > angle_thresh:
            return False
        # For horizontal text (angle near 0)
        if abs(box1.angle) < angle_thresh:
            # Check if vertical centers are close
            y1 = box1.y + box1.height / 2
            y2 = box2.y + box2.height / 2
            avg_height = (box1.height + box2.height) / 2
            ret = abs(y1 - y2) < avg_height * line_thresh
        else:
            # For rotated text, project centers onto the perpendicular direction
            import math

            theta = math.radians(box1.angle)
            # Direction perpendicular to text line
            dx = -math.sin(theta)
            dy = math.cos(theta)
            c1x = box1.x + box1.width / 2
            c1y = box1.y + box1.height / 2
            c2x = box2.x + box2.width / 2
            c2y = box2.y + box2.height / 2
            # Project difference onto perpendicular direction
            perp_dist = abs((c2x - c1x) * dx + (c2y - c1y) * dy)
            avg_dim = (box1.height + box2.height) / 2
            ret = perp_dist < avg_dim * line_thresh
        return ret

    @staticmethod
    def merge_overlapping_boxes(
        boxes: list["Box"],
        iou_threshold: float = 0.3,
        angle_thresh: float = 10.0,
        line_thresh: float = 0.5,
    ) -> list["Box"]:
        """
        Merge all overlapping boxes in a list using a greedy algorithm,
        but only if they are on the same line and have similar angle.
        """
        merged = []
        used = [False] * len(boxes)
        for i, box in enumerate(boxes):
            if used[i]:
                continue
            curr = box
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                if Box.iou(curr, boxes[j]) > iou_threshold and Box.is_on_same_line(
                    curr,
                    boxes[j],
                    angle_thresh=angle_thresh,
                    line_thresh=line_thresh,
                ):
                    curr = Box.merge(curr, boxes[j])
                    used[j] = True
            merged.append(curr)
            used[i] = True
        return merged


register_type(Box, Box.get_schema)
