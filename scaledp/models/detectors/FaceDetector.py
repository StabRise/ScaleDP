from types import MappingProxyType

from scaledp.enums import Device
from scaledp.models.detectors.YoloOnnxDetector import YoloOnnxDetector


class FaceDetector(YoloOnnxDetector):
    """Face detector using YOLO ONNX model."""

    defaultParams = MappingProxyType(
        {
            "inputCol": "image",
            "outputCol": "boxes",
            "keepInputData": False,
            "scaleFactor": 1.0,
            "scoreThreshold": 0.2,
            "device": Device.CPU,
            "batchSize": 2,
            "partitionMap": False,
            "numPartitions": 0,
            "pageCol": "",
            "pathCol": "",
            "propagateError": False,
            "task": "detect",
            "onlyRotated": False,
            "model": "StabRise/face_detection",
            "labels": ["face"],
        },
    )
