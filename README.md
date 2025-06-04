
<p align="center">
  <br/>
    <a href="https://stabrise.com/scaledp/" target="_blank"><img alt="ScaleDP" src="https://stabrise.com/static/images/projects/scaledp.webp" width="450" style="max-width: 100%;"></a>
  <br/>
</p>

<p align="center">
    <i>An Open-Source Library for Processing Documents using AI/ML in Apache Spark.</i>
</p>

<p align="center">
    <a href="https://pypi.org/project/scaledp/" alt="Package on PyPI"><img src="https://img.shields.io/pypi/v/scaledp.svg" /></a>
    <a href="https://github.com/stabrise/spark-pdf/blob/main/LICENSE"><img alt="GitHub" src="https://img.shields.io/github/license/stabrise/spark-pdf.svg?color=blue"></a>
    <a href="https://stabrise.com"><img alt="StabRise" src="https://img.shields.io/badge/powered%20by-StabRise-orange.svg?style=flat&colorA=E1523D&colorB=007D8A"></a>
    <a href="https://app.codacy.com/gh/StabRise/ScaleDP/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade">
    <img src="https://app.codacy.com/project/badge/Grade/98570508281140c2a33e616a4f749c20" alt="Codacy Badge" />
</a></p>

---

**Source Code**: <a href="https://github.com/StabRise/ScaleDP/" target="_blank">https://github.com/StabRise/ScaleDP</a>

**Quickstart**: <a href="https://colab.research.google.com/github/StabRise/scaledp-tutorials/blob/master/1.QuickStart.ipynb" target="_blank">1.QuickStart.ipynb</a>

**Tutorials**: <a href="https://github.com/StabRise/ScaleDP-Tutorials/" target="_blank">https://github.com/StabRise/ScaleDP-Tutorials</a>

---

# Welcome to the ScaleDP library

ScaleDP is library allows you to process documents using AI/ML capabilities and scale it using Apache Spark.

**LLM** (Large Language Models) and **VLM** (Vision Language Models) models are used to extract data from text and images in combination with **OCR** engines.

Discover pre-trained models for your projects or play with the thousands of models hosted on the [Hugging Face Hub](https://huggingface.co/).

## Key features

### Document processing:

- ✅ Loading PDF documents/Images to the Spark DataFrame (using [Spark PDF Datasource](https://github.com/stabrise/spark-pdf) and as `binaryFile`)
- ✅ Extraction text/images from PDF documents/Images
- ✅ Zero-Shot extraction **structured data** from text/images using LLM and ML models
- ✅ Possibility run as REST API service without Spark Session for have minimum processing latency
- ✅ Support Streaming mode for processing documents in real-time

### LLM:

Support OpenAI compatible API for call LLM/VLM models (GPT, Gemini, GROQ, etc.)

- OCR Images/PDF documents using Vision LLM models
- Extract data from the image using Vision LLM models
- Extract data from the text/images using LLM models
- Extract data using DSPy framework
- NER using LLM's
- Visualize results

### NLP:

- Extract data from the text/images using NLP models from the Hugging Face Hub
- NER using classical ML models

### OCR:

Support various open-source OCR engines:

 - [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) 
 - [Easy OCR](https://github.com/JaidedAI/EasyOCR)   
 - [Surya OCR](https://github.com/VikParuchuri/surya) 
 - [DocTR](https://github.com/mindee/doctr)
 - Vision LLM models

### CV:
- Object detection on images using YOLO models
- Text detection on images


## Installation

### Prerequisites

- Python 3.10 or higher
- Apache Spark 3.5 or higher
- Java 8

### Installation using pip

Install the `ScaleDP` package with [pip](https://pypi.org/project/scaledp/):

```bash
pip install scaledp
```

### Installation using Docker

Build image:

```bash
  docker build -t scaledp .
```

Run container:
```bash
  docker run -p 8888:8888 scaledp:latest
```

Open Jupyter Notebook in your browser:
```bash
  http://localhost:8888
```

## Qiuckstart

Start a Spark session with ScaleDP:

```python
from scaledp import *
spark = ScaleDPSession()
spark
```

Read example image file:

```python
image_example = files('resources/images/Invoice.png')
df = spark.read.format("binaryFile") \
    .load(image_example)

df.show_image("content")
```
Output:

<img src="https://github.com/StabRise/ScaleDP/blob/master/images/ImageOutput.png?raw=true" width="400">


## Zero-Shot data Extraction from the Image:

```python
from pydantic import BaseModel
import json

class Items(BaseModel):
    date: str
    item: str
    note: str
    debit: str

class InvoiceSchema(BaseModel):
    hospital: str
    tax_id: str
    address: str
    email: str
    phone: str
    items: list[Items]
    total: str
    

pipeline = PipelineModel(stages=[
    DataToImage(
        inputCol="content",
        outputCol="image"
    ),
    LLMVisualExtractor(
        inputCol="image",
        outputCol="invoice",
        model="gemini-1.5-flash",
        apiKey="",
        apiBase="https://generativelanguage.googleapis.com/v1beta/",
        schema=json.dumps(InvoiceSchema.model_json_schema())
    )
])

result = pipeline.transform(df).cache()
```

Show the extracted json:

```python
result.show_json("invoice")
```

<img src="https://github.com/StabRise/ScaleDP/blob/master/images/LLMVisualExtractorJson.png?raw=true" width="400">

Let's show Invoice as Structured Data in Data Frame

```python
result.select("invoice.data.*").show()
```

Output:

```text
+-------------------+---------+--------------------+--------------------+--------------+--------------------+-------+
|           hospital|   tax_id|             address|               email|         phone|               items|  total|
+-------------------+---------+--------------------+--------------------+--------------+--------------------+-------+
|Hope Haven Hospital|26-123123|855 Howard Street...|hopedutton@hopeha...|(123) 456-1238|[{10/21/2022, App...|1024.50|
+-------------------+---------+--------------------+--------------------+--------------+--------------------+-------+
```

Schema:

```python
result.printSchema()
```

```text
root
 |-- path: string (nullable = true)
 |-- modificationTime: timestamp (nullable = true)
 |-- length: long (nullable = true)
 |-- image: struct (nullable = true)
 |    |-- path: string (nullable = false)
 |    |-- resolution: integer (nullable = false)
 |    |-- data: binary (nullable = false)
 |    |-- imageType: string (nullable = false)
 |    |-- exception: string (nullable = false)
 |    |-- height: integer (nullable = false)
 |    |-- width: integer (nullable = false)
 |-- invoice: struct (nullable = true)
 |    |-- path: string (nullable = false)
 |    |-- json_data: string (nullable = true)
 |    |-- type: string (nullable = false)
 |    |-- exception: string (nullable = false)
 |    |-- processing_time: double (nullable = false)
 |    |-- data: struct (nullable = true)
 |    |    |-- hospital: string (nullable = false)
 |    |    |-- tax_id: string (nullable = false)
 |    |    |-- address: string (nullable = false)
 |    |    |-- email: string (nullable = false)
 |    |    |-- phone: string (nullable = false)
 |    |    |-- items: array (nullable = false)
 |    |    |    |-- element: struct (containsNull = false)
 |    |    |    |    |-- date: string (nullable = false)
 |    |    |    |    |-- item: string (nullable = false)
 |    |    |    |    |-- note: string (nullable = false)
 |    |    |    |    |-- debit: string (nullable = false)
 |    |    |-- total: string (nullable = false)
```

## NER using model from the HuggingFace models Hub

Define pipeline for extract text from the image and run NER:

```python
pipeline = PipelineModel(stages=[
    DataToImage(inputCol="content", outputCol="image"),
    TesseractOcr(inputCol="image", outputCol="text", psm=PSM.AUTO, keepInputData=True),
    Ner(model="obi/deid_bert_i2b2", inputCol="text", outputCol="ner", keepInputData=True),
    ImageDrawBoxes(inputCols=["image", "ner"], outputCol="image_with_boxes", lineWidth=3, 
                   padding=5, displayDataList=['entity_group'])
])

result = pipeline.transform(df).cache()

result.show_text("text")
```

Output:

<img src="https://github.com/StabRise/ScaleDP/blob/master/images/TextOutput.png?raw=true" width="400">

Show NER results:

```python
result.show_ner(limit=20)
```

Output:
```text
+------------+-------------------+----------+-----+---+--------------------+
|entity_group|              score|      word|start|end|               boxes|
+------------+-------------------+----------+-----+---+--------------------+
|        HOSP|  0.991257905960083|  Hospital|    0|  8|[{Hospital:, 0.94...|
|         LOC|  0.999171257019043|    Dutton|   10| 16|[{Dutton,, 0.9609...|
|         LOC| 0.9992585778236389|        MI|   18| 20|[{MI, 0.93335297,...|
|          ID| 0.6838774085044861|        26|   29| 31|[{26-123123, 0.90...|
|       PHONE| 0.4669836759567261|         -|   31| 32|[{26-123123, 0.90...|
|       PHONE| 0.7790696024894714|    123123|   32| 38|[{26-123123, 0.90...|
|        HOSP|0.37445762753486633|      HOPE|   39| 43|[{HOPE, 0.9525460...|
|        HOSP| 0.9503226280212402|     HAVEN|   44| 49|[{HAVEN, 0.952546...|
|         LOC| 0.9975488185882568|855 Howard|   59| 69|[{855, 0.94682700...|
|         LOC| 0.9984399676322937|    Street|   70| 76|[{Street, 0.95823...|
|        HOSP| 0.3670221269130707|  HOSPITAL|   77| 85|[{HOSPITAL, 0.959...|
|         LOC| 0.9990363121032715|    Dutton|   86| 92|[{Dutton,, 0.9647...|
|         LOC|  0.999313473701477|  MI 49316|   94|102|[{MI, 0.94589012,...|
|       PHONE| 0.9830010533332825|   ( 123 )|  110|115|[{(123), 0.595334...|
|       PHONE| 0.9080978035926819|       456|  116|119|[{456-1238, 0.955...|
|       PHONE| 0.9378324151039124|         -|  119|120|[{456-1238, 0.955...|
|       PHONE| 0.8746233582496643|      1238|  120|124|[{456-1238, 0.955...|
|     PATIENT|0.45354968309402466|hopedutton|  132|142|[{hopedutton@hope...|
|       EMAIL|0.17805588245391846| hopehaven|  143|152|[{hopedutton@hope...|
|        HOSP|  0.505658745765686|   INVOICE|  157|164|[{INVOICE, 0.9661...|
+------------+-------------------+----------+-----+---+--------------------+
```

Visualize NER results:

```python
result.visualize_ner(labels_list=["DATE", "LOC"])
```
<img src="https://github.com/StabRise/ScaleDP/blob/master/images/NerVisual.png?raw=true" width="400">

Original image with NER results:

```python
result.show_image("image_with_boxes")
```
<img src="https://github.com/StabRise/ScaleDP/blob/master/images/NerVisualOnImage.png?raw=true" width="400">

## Ocr engines

|                   | Bbox  level | Support GPU | Separate model  for text detection | Processing time 1 page (CPU/GPU) secs | Support Handwritten Text |
|-------------------|-------------|-------------|------------------------------------|---------------------------------------|--------------------------|
| [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)     | character   | no          | no                                 | 0.2/no                                | not good                 |
| Tesseract OCR CLI | character   | no          | no                                 | 0.2/no                                | not good                 |
| [Easy OCR](https://github.com/JaidedAI/EasyOCR)          | word        | yes         | yes                                |                                       |                          |
| [Surya OCR](https://github.com/VikParuchuri/surya)         | line        | yes         | yes                                |                                       |                          |
| [DocTR](https://github.com/mindee/doctr)       | word        | yes         | yes                                |                                       |                          |


## Projects based on the ScaleDP

 - [PDF Redaction](https://pdf-redaction.com/) - Free AI-powered tool for redact PDF files (remove sensitive information) online.


<a href="https://pdf-redaction.com/"><img alt="pdf-redaction" src="https://media.licdn.com/dms/image/v2/D4D22AQGhRpexOnAbyA/feedshare-shrink_800/B4DZVmbKWPHIAg-/0/1741180153002?e=1744243200&v=beta&t=lRQXyJ5nHYvdU4uF6LJuq69oKs72yPBs1xts2IrJgxc"/></a>


## Disclaimer

This project is not affiliated with, endorsed by, or connected to the Apache Software Foundation or Apache Spark.
