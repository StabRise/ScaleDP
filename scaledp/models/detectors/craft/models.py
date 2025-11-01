import onnxruntime
from crafter.resources import res


class Craftnet:
    def __init__(self, onnx_path=None):

        session_options = onnxruntime.SessionOptions()
        if onnx_path is None:
            onnx_path = res("craftnet.onnx")
        self._onnx_session = onnxruntime.InferenceSession(
            onnx_path,
            sess_options=session_options,
        )

    def __call__(self, image):
        return self._onnx_session.run(None, {"image": image})


class Refinenet:
    def __init__(self, onnx_path=None):
        if onnx_path is None:
            onnx_path = res("refinenet.onnx")
        self._onnx_session = onnxruntime.InferenceSession(onnx_path)

    def __call__(self, y, feature):
        return self._onnx_session.run(None, {"y": y, "feature": feature})[0]
