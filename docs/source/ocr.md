OCR models
==========

The OCR models in ScaledP provide robust optical character recognition capabilities for extracting text from images and documents. These models leverage advanced deep learning techniques to deliver high accuracy and performance across various use cases.

## Available OCR Engines

End-to-end OCR solutions available in ScaledP include:

- **Tesseract OCR**: An open-source OCR engine that supports multiple languages and is widely used for text extraction tasks.
- **DocTR OCR**: A deep learning-based OCR model that offers superior accuracy, especially for complex documents and layouts.
- **Surya OCR**: A high-performance OCR model optimized for speed and accuracy, suitable for real-time applications.
- **EasyOCR**: A lightweight OCR model that provides fast text recognition with support for multiple languages.
- **LLMOcr**: An OCR that utilizes large language models to enhance text recognition capabilities.

## Text Detectors

Text detectors are used to identify and locate text regions within images.
In some cases useful to run it as separate step in the OCR pipeline.

See the following text detectors available in ScaledP:

* [**CraftTextDetector**](#CraftTextDetector)
* [**DBNetOnnxDetector**](#DBNetOnnxDetector)
* **YoloOnnxTextDetector**
* **DocTRTextDetector**

## Text Recognizers

Text recognizers can recognize text from images contains single 
line/word/character. 
Available text recognizers in ScaledP include:

* [**TesseractRecognizer**](#TesseractRecognizer)
