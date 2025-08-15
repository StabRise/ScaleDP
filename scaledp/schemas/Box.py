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
        return abs(self.angle) >= 10


register_type(Box, Box.get_schema)
