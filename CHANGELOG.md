## [0.2.5] - 10.11.2025

### 🚀 Features

- Added param 'returnEmpty' to [ImageCropBoxes](https://scaledp.stabrise.com/en/latest/image/image_crop_boxes.html) for avoid to have exceptions if no boxes are found
- Added labels param to the [YoloOnnxDetector](https://scaledp.stabrise.com/en/latest/models/detectors/yolo_onnx_detector.html)
- Improve displaying labels in [ImageDrawBoxes](https://scaledp.stabrise.com/en/latest/image/image_draw_boxes.html)

### 🧰 Maintenance
- Updated versions of dependencies (Pandas, Numpy, OpenCV)

### 🐛 Bug Fixes

- Fixed convert color schema in [YoloOnnxDetector](https://scaledp.stabrise.com/en/latest/models/detectors/yolo_onnx_detector.html)
- Fixed show utils on Google Colab
- Fixed imports of the DataFrame

### 📘 Jupyter Notebooks

- [YoloOnnxDetector.ipynb](https://github.com/StabRise/ScaleDP-Tutorials/blob/master/object-detection/1.YoloOnnxDetector.ipynb)
- [FaceDetection.ipynb](https://github.com/StabRise/ScaleDP-Tutorials/blob/master/object-detection/2.FaceDetection.ipynb)
- [SignatureDetection.ipynb](https://github.com/StabRise/ScaleDP-Tutorials/blob/master/object-detection/3.SignatureDetection.ipynb)

### 📝 Blog Posts

- [Running YOLO Models on Spark Using ScaleDP](https://stabrise.com/blog/running_yolo_on_spark_with_scaledp/)


## [0.2.4] - 02.11.2025

### 🚀 Features

- Added [FaceDetector](https://scaledp.stabrise.com/en/latest/models/detectors/face_detector.html) transformer
- Added [SignatureDetector](https://scaledp.stabrise.com/en/latest/models/detectors/signature_detector.html) transformers
- Added [PdfAssembler](https://scaledp.stabrise.com/en/latest/pdf/pdf_assembler.html) transformer for assembling PDFs
- Updated [ImageCropBoxes](https://scaledp.stabrise.com/en/latest/image/image_crop_boxes.html) to support multiple boxes
- Added LineOrientation detector model to the [TesseractRecognizer](https://scaledp.stabrise.com/en/latest/models/recognizers/tesseract_recognizer.html)
- Added possibility to use subfields in [Show Utils](https://scaledp.stabrise.com/en/latest/show_utils.html)
- Added padding option to [YoloOnnxDetector](https://scaledp.stabrise.com/en/latest/models/detectors/yolo_onnx_detector.html)

### 🐛 Bug Fixes

- Fixed borders in [Show Utils](https://scaledp.stabrise.com/en/latest/show_utils.html)
