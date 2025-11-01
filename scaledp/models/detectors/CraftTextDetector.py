from types import MappingProxyType
from typing import Any

import numpy
from pyspark import keyword_only

from scaledp.enums import Device
from scaledp.models.detectors.BaseDetector import BaseDetector
from scaledp.params import HasBatchSize, HasDevice, Param, Params, TypeConverters
from scaledp.schemas.Box import Box
from scaledp.schemas.DetectorOutput import DetectorOutput


class CraftTextDetector(BaseDetector, HasDevice, HasBatchSize):
    """CRAFT text detector."""

    _craft_net = None
    _refine_net = None

    defaultParams = MappingProxyType(
        {
            "inputCol": "image",
            "outputCol": "boxes",
            "keepInputData": False,
            "scaleFactor": 1.0,
            "scoreThreshold": 0.7,
            "textThreshold": 0.4,
            "linkThreshold": 0.4,
            "sizeThreshold": -1,
            "width": 1280,
            "withRefiner": False,
            "device": Device.CPU,
            "batchSize": 2,
            "partitionMap": False,
            "numPartitions": 0,
            "pageCol": "page",
            "pathCol": "path",
            "propagateError": False,
            "onlyRotated": False,
        },
    )

    textThreshold = Param(
        Params._dummy(),
        "textThreshold",
        "Threshold for text region score",
        typeConverter=TypeConverters.toFloat,
    )

    linkThreshold = Param(
        Params._dummy(),
        "linkThreshold",
        "Threshold for link affinity score",
        typeConverter=TypeConverters.toFloat,
    )

    sizeThreshold = Param(
        Params._dummy(),
        "sizeThreshold",
        "Threshold for height of detected regions",
        typeConverter=TypeConverters.toInt,
    )

    width = Param(
        Params._dummy(),
        "width",
        "Width for image resizing",
        typeConverter=TypeConverters.toInt,
    )

    withRefiner = Param(
        Params._dummy(),
        "withRefiner",
        "Enable refiner network postprocessing",
        typeConverter=TypeConverters.toBoolean,
    )

    @keyword_only
    def __init__(self, **kwargs: Any) -> None:
        super(CraftTextDetector, self).__init__()
        self._setDefault(**self.defaultParams)
        self._set(**kwargs)
        self.get_model({k.name: v for k, v in self.extractParamMap().items()})

    @classmethod
    def get_model(cls, params):
        if cls._craft_net and cls._refine_net:
            return cls._craft_net, cls._refine_net

        from craft_text_detector import load_craftnet_model, load_refinenet_model

        device = "cuda" if int(params["device"]) == Device.CUDA.value else "cpu"
        use_cuda = device == "cuda"

        craft_net = load_craftnet_model(
            cuda=use_cuda,
        )

        refine_net = None
        if params.get("withRefiner"):
            refine_net = load_refinenet_model(
                cuda=use_cuda,
            )

        cls._craft_net = craft_net
        cls._refine_net = refine_net
        return craft_net, refine_net

    @classmethod
    def call_detector(cls, images, params):
        import cv2
        from craft_text_detector import craft_utils, image_utils, torch_utils

        craft_net, refine_net = cls.get_model(params)
        use_cuda = int(params["device"]) == Device.CUDA.value
        results = []

        for image, image_path in images:
            try:
                # Convert PIL to OpenCV format

                img_cv = image_utils.read_image(numpy.array(image)[:, :, ::-1])

                # Resize
                img_resized, target_ratio, size_heatmap = (
                    image_utils.resize_aspect_ratio(
                        img_cv,
                        params["width"],
                        interpolation=cv2.INTER_LINEAR,
                    )
                )
                ratio_h = ratio_w = 1 / target_ratio

                # Preprocess
                x = image_utils.normalizeMeanVariance(img_resized)
                x = torch_utils.from_numpy(x).permute(2, 0, 1)
                x = torch_utils.Variable(x.unsqueeze(0))
                if use_cuda:
                    x = x.cuda()

                # Forward pass
                with torch_utils.no_grad():
                    y, feature = craft_net(x)

                score_text = y[0, :, :, 0].cpu().data.numpy()
                score_link = y[0, :, :, 1].cpu().data.numpy()

                # Refine if enabled
                if refine_net is not None:
                    with torch_utils.no_grad():
                        y_refiner = refine_net(y, feature)
                    score_link = y_refiner[0, :, :, 0].cpu().data.numpy()

                # Post-process
                boxes, _ = craft_utils.getDetBoxes(
                    score_text,
                    score_link,
                    params["scoreThreshold"],
                    params["linkThreshold"],
                    params["textThreshold"],
                    poly=False,
                )

                # Adjust coordinates
                boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)

                # Convert to Box objects
                box_objects = [Box.from_polygon(box) for box in boxes]

                results.append(
                    DetectorOutput(
                        path=image_path,
                        type="craft",
                        bboxes=box_objects,
                        exception="",
                    ),
                )

            except Exception as e:
                raise e
                results.append(
                    DetectorOutput(
                        path=image_path,
                        type="craft",
                        bboxes=[],
                        exception=f"CraftTextDetector error: {e!s}",
                    ),
                )

        return results

    @classmethod
    def call_detector1(cls, images, params):
        import cv2
        from crafter import craft_utils, image_utils

        craft_net, refine_net = cls.get_model(params)
        results = []

        for image, image_path in images:
            try:
                # Convert PIL to OpenCV format

                img_cv = image_utils.read_image(numpy.array(image)[:, :, ::-1])

                # Resize
                img_resized, target_ratio, size_heatmap = (
                    image_utils.resize_aspect_ratio(
                        img_cv,
                        params["width"],
                        interpolation=cv2.INTER_LINEAR,
                    )
                )
                ratio_h = ratio_w = 1 / target_ratio

                # Preprocess
                x = image_utils.normalizeMeanVariance(img_resized)
                x = numpy.transpose(x, (2, 0, 1))  # [h, w, c] to [c, h, w]
                x = numpy.expand_dims(x, 0)  # [c, h, w] to [b, c, h, w]

                # Forward pass
                y, feature = craft_net(x)

                score_text = y[0, 0, :, :]
                score_link = y[0, 1, :, :]

                # Refine if enabled
                if refine_net is not None:
                    y_refiner = refine_net(y, feature)
                    score_link = y_refiner[0, 0, :, :]

                # Post-process
                boxes, _ = craft_utils.getDetBoxes(
                    score_text,
                    score_link,
                    params["scoreThreshold"],
                    params["linkThreshold"],
                    params["textThreshold"],
                    poly=False,
                )

                # Adjust coordinates
                boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)

                # Convert to Box objects
                box_objects = [Box.from_polygon(box) for box in boxes]

                results.append(
                    DetectorOutput(
                        path=image_path,
                        type="craft",
                        bboxes=box_objects,
                        exception="",
                    ),
                )

            except Exception as e:
                raise e
                results.append(
                    DetectorOutput(
                        path=image_path,
                        type="craft",
                        bboxes=[],
                        exception=f"CraftTextDetector error: {e!s}",
                    ),
                )

        return results

    def setTextThreshold(self, value):
        return self._set(textThreshold=value)

    def setLinkThreshold(self, value):
        return self._set(linkThreshold=value)

    def setSizeThreshold(self, value):
        return self._set(sizeThreshold=value)

    def setWidth(self, value):
        return self._set(width=value)

    def setWithRefiner(self, value):
        return self._set(withRefiner=value)
