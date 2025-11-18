from types import MappingProxyType

from scaledp.enums import Device
from scaledp.models.detectors.YoloOnnxDetector import YoloOnnxDetector


class SignatureDetector(YoloOnnxDetector):
    defaultParams = MappingProxyType(
        {
            "inputCol": "image",
            "outputCol": "signatures",
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
            "model": "StabRise/signature_detection",
            "labels": ["signature"],
        },
    )
