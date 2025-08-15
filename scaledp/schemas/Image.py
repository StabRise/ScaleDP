import io
import logging
from dataclasses import dataclass

import imagesize
from PIL import Image as pImage

from scaledp.utils.dataclass import BinaryT, map_dataclass_to_struct, register_type

from ..enums import ImageType


@dataclass(order=True)
class Image(object):
    """Image object for represent image data in Spark Dataframe."""

    path: str
    resolution: int = 0
    data: BinaryT = bytes()
    imageType: str = ImageType.FILE.value
    exception: str = ""
    height: int = 0
    width: int = 0

    def to_pil(self) -> pImage.Image:
        """Convert image to PIL Image format."""
        if self.imageType in (ImageType.FILE.value, ImageType.WEBP.value):
            return pImage.open(io.BytesIO(self.data))
        raise ValueError("Invalid image type.")

    def to_cv2(self):
        """Convert image to OpenCV format."""
        import cv2
        import numpy as np

        if self.imageType in (ImageType.FILE.value, ImageType.WEBP.value):
            # Convert bytes to numpy array
            nparr = np.frombuffer(self.data, np.uint8)
            # Decode the numpy array as an image
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        raise ValueError("Invalid image type.")

    def to_io_stream(self) -> io.BytesIO:
        return io.BytesIO(self.data)

    def to_webp(self) -> "Image":
        if self.imageType == ImageType.FILE.value:
            image = pImage.open(io.BytesIO(self.data))
            buff = io.BytesIO()
            image.save(buff, "webp")
            self.data = buff.getvalue()
        return self

    @staticmethod
    def from_binary(data, path, imageType, resolution=None, width=None, height=None):
        img = Image(
            path=path,
            data=data,
            imageType=ImageType.FILE.value,
            resolution=resolution,
        )
        if data is None or len(data) == 0:
            raise ValueError("Empty image data.")
        if imageType in (ImageType.FILE.value, ImageType.WEBP.value):
            if height is not None:
                img.height = height
            if width is not None:
                img.width = width
            if width is None and height is None:
                img.width, img.height = imagesize.get(io.BytesIO(img.data))
                if img.width == -1:
                    raise Exception("Unable to read image.")
                logging.info(f"Image size: {img.width}x{img.height}")
        return img

    @staticmethod
    def from_pil(data, path, imageType, resolution):
        buff = io.BytesIO()
        if imageType == ImageType.WEBP.value:
            data.save(buff, "webp")
        else:
            data.save(buff, "png")
        return Image(
            path=path,
            data=buff.getvalue(),
            imageType=ImageType.FILE.value,
            width=data.width,
            height=data.height,
            resolution=resolution,
        )

    @staticmethod
    def get_schema():
        return map_dataclass_to_struct(Image)

    def __str__(self) -> str:
        return (
            f"Image(path={self.path}, resolution={self.resolution}, "
            f"imageType={self.imageType}, exception={self.exception}, "
            f"height={self.height}, width={self.width})"
        )

    def __repr__(self) -> str:
        return self.__str__()


register_type(Image, Image.get_schema)
