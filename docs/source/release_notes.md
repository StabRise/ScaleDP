Release Notes
=============

This document outlines the release notes for the ScaledP project. It includes information about new features, bug fixes, and other changes made in each version.


## [unreleased]

### 🚀 Features

- Added [TextEmbeddings](#TextEmbeddings) transformer, for compute embedding using SentenceTransformers


## [0.2.6] - 19.11.2025

### 🚀 Features

- Enable support GPU in [YoloOnnxDetector](#YoloOnnxDetector)

### 🔄 Updates

- Change default values for detectors

### 📘 Jupyter Notebooks

- [YoloOnnxDetectorBenchamrks.ipynb](https://github.com/StabRise/ScaleDP-Tutorials/blob/master/object-detection/4.YoloOnnxDetectorBenchmarks.ipynb) - Benchmarking YOLO model with different parameters configurations on CPU and GPU


## [0.2.5] - 10.11.2025

### 🚀 Features

- Added param 'returnEmpty' to [ImageCropBoxes](#ImageCropBoxes) for avoid to have exceptions if no boxes are found
- Added labels param to the [YoloOnnxDetector](#YoloOnnxDetector)
- Improve displaying labels in [ImageDrawBoxes](#ImageDrawBoxes)

### 🧰 Maintenance
- Updated versions of dependencies (Pandas, Numpy, OpenCV)

### 🐛 Bug Fixes

- Fixed convert color schema in [YoloOnnxDetector](#YoloOnnxDetector)
- Fixed show utils on Google Colab
- Fixed imports of the DataFrame

### 📘 Jupyter Notebooks

- [YoloOnnxDetector.ipynb](https://github.com/StabRise/ScaleDP-Tutorials/blob/master/object-detection/1.YoloOnnxDetector.ipynb)
- [FaceDetection.ipynb](https://github.com/StabRise/ScaleDP-Tutorials/blob/master/object-detection/2.FaceDetection.ipynb)
- [SignatureDetection.ipynb](https://github.com/StabRise/ScaleDP-Tutorials/blob/master/object-detection/3.SignatureDetection.ipynb)

### 📝 Blog Posts

- [Running YOLO Models on Spark Using ScaleDP](https://stabrise.com/blog/running_yolo_on_spark_with_scaledp/)


## 0.2.4 - 02.11.2025

### 🚀 Features

- Added <project:#FaceDetector>, <project:#SignatureDetector>
- Updated [ImageCropBoxes](#ImageCropBoxes) to support multiple boxes
- Added LineOrientation detector model to the [TesseractRecognizer](#TesseractRecognizer)
- Added [PdfAssembler](#PdfAssembler) transformer for assembling PDFs
- Added possibility to use subfields in show utils
- Added padding option to the [YoloOnnxDetector](#YoloOnnxDetector)

### 🐛 Bug Fixes

- Fixed borders in [show utils](#ShowUtils)
