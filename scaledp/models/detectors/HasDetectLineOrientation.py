from pathlib import Path
from typing import Any, ClassVar

import cv2
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from pyspark.ml.param import Param, Params, TypeConverters


class HasDetectLineOrientation(Params):
    """
    Mixin for param detectLineOrientation: whether to detect line orientation.
    and logic for orientation detection.
    """

    detectLineOrientation = Param(
        Params._dummy(),
        "detectLineOrientation",
        "Whether to detect line orientation.",
        typeConverter=TypeConverters.toBoolean,
    )

    oriModel = Param(
        Params._dummy(),
        "oriModel",
        "Text line Orientation Model.",
        typeConverter=TypeConverters.toString,
    )

    def getOriModel(self) -> str:
        """
        Gets the value of model or its default value.
        """
        return self.getOrDefault(self.model)

    def setOriModel(self, value: str) -> Any:
        """
        Sets the value of :py:attr:`model`.
        """
        return self._set(model=value)

    _orientation_session: ClassVar = None
    _orientation_input_name: ClassVar = None
    _orientation_label_list: ClassVar = ["0_degree", "180_degree"]

    def setDetectLineOrientation(self, value: bool):
        return self._set(detectLineOrientation=value)

    def getDetectLineOrientation(self) -> bool:
        return self.getOrDefault(self.detectLineOrientation)

    @classmethod
    def _load_orientation_model(cls, params):
        if cls._orientation_session is None:
            model_path = params.get("oriModel")
            if not Path(model_path).is_file():
                model = hf_hub_download(repo_id=model_path, filename="model.onnx")
            cls._orientation_session = ort.InferenceSession(model)
            cls._orientation_input_name = cls._orientation_session.get_inputs()[0].name

    @staticmethod
    def _preprocess_for_orientation(pil_img):
        img = np.array(pil_img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (160, 80))  # width, height
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = np.expand_dims(img, 0)  # batch
        return img.astype(np.float32)

    @classmethod
    def detect_orientation(cls, pil_img, params):
        """Detects orientation (0 or 180 degrees) of a PIL image."""
        cls._load_orientation_model(params)
        inp = cls._preprocess_for_orientation(pil_img)
        outputs = cls._orientation_session.run(None, {cls._orientation_input_name: inp})
        pred_idx = np.argmax(outputs[0], axis=1)[0]
        pred_label = cls._orientation_label_list[pred_idx]
        return pred_label

    @classmethod
    def auto_orient_image(cls, pil_img, params):
        """Rotates the image to 0 degrees if needed."""
        orientation = cls.detect_orientation(pil_img, params)
        if orientation == "180_degree":
            return pil_img.rotate(180, expand=True), orientation
        return pil_img, orientation
