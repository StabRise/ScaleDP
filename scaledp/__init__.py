import os
import sys
from importlib import resources
from importlib.util import find_spec
from pathlib import Path

import pyspark
from pyspark.ml.pipeline import PipelineModel
from pyspark.sql import DataFrame, SparkSession

from scaledp import enums
from scaledp.image.DataToImage import DataToImage
from scaledp.image.ImageCropBoxes import ImageCropBoxes
from scaledp.image.ImageDrawBoxes import ImageDrawBoxes
from scaledp.models.detectors.DocTRTextDetector import DocTRTextDetector
from scaledp.models.detectors.YoloDetector import YoloDetector
from scaledp.models.extractors.DSPyExtractor import DSPyExtractor
from scaledp.models.extractors.LLMExtractor import LLMExtractor
from scaledp.models.extractors.LLMVisualExtractor import LLMVisualExtractor
from scaledp.models.ner.LLMNer import LLMNer
from scaledp.models.ner.Ner import Ner
from scaledp.models.recognizers.DocTROcr import DocTROcr
from scaledp.models.recognizers.EasyOcr import EasyOcr
from scaledp.models.recognizers.LLMOcr import LLMOcr
from scaledp.models.recognizers.SuryaOcr import SuryaOcr
from scaledp.models.recognizers.TesseractOcr import TesseractOcr
from scaledp.models.recognizers.TesseractRecognizer import TesseractRecognizer
from scaledp.pdf.PdfDataToImage import PdfDataToImage
from scaledp.text.TextToDocument import TextToDocument
from scaledp.utils.show_utils import (
    show_image,
    show_json,
    show_ner,
    show_pdf,
    show_text,
    visualize_ner,
)

DataFrame.show_image = (
    lambda self, column="", limit=5, width=None, show_meta=True: show_image(
        self,
        column,
        limit,
        width,
        show_meta,
    )
)
DataFrame.show_pdf = (
    lambda self, column="", limit=5, width=None, show_meta=True: show_pdf(
        self,
        column,
        limit,
        width,
        show_meta,
    )
)
DataFrame.show_ner = lambda self, column="ner", limit=20, truncate=True: show_ner(
    self,
    column,
    limit,
    truncate,
)
DataFrame.show_text = (
    lambda self, column="", field="text", limit=20, width=None: show_text(
        self,
        column,
        field,
        limit,
        width,
    )
)
DataFrame.show_json = (
    lambda self, column="", field="data", limit=20, width=None: show_json(
        self,
        column,
        field,
        limit,
        width,
    )
)
DataFrame.visualize_ner = (
    lambda self, column="ner", text_column="text", limit=20, width=None: visualize_ner(
        self,
        column,
        text_column,
        limit,
        width,
    )
)


def version():
    with (Path.parent / "VERSION").open(encoding="utf-8") as version_file:
        return version_file.read().strip()


__version__ = version()


def files(path):
    """File resources."""
    return resources.files("scaledp").joinpath(path).as_posix()


def aws_version():
    spark_hadoop_map = {
        "3.0": "2.7.4",
        "3.1": "3.2.0",
        "3.2": "3.3.1",
        "3.3": "3.3.2",
        "3.4": "3.3.4",
        "3.5": "3.3.4",
    }
    return spark_hadoop_map[pyspark.__version__[:3]]


def ScaleDPSession(
    conf=None,
    master_url="local[*]",
    with_aws=False,
    with_pro=False,
    logLevel="ERROR",
):
    """
    Start Spark session with ScaleDP.

    @param conf: Instance of SparkConf or dict with extra configuration.
    @param master_url: Spark master URL
    @param with_aws: Enable AWS support
    @param logLevel: Log level
    """
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["TRANSFORMERS_VERBOSITY"] = logLevel.lower()

    if with_pro and find_spec("scaledp_pro") is None:
        raise ImportError(
            "ScaleDP Pro is not installed. Please install it using 'pip install scaledp-pro'",
        )

    jars = []
    jars_packages = []
    default_conf = {
        "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
        "spark.kryoserializer.buffer.max": "200M",
        "spark.driver.memory": "8G",
    }

    if with_aws:
        jars_packages.append("org.apache.hadoop:hadoop-aws:" + aws_version())

    if conf:
        if not isinstance(conf, dict):
            conf = dict(conf.getAll())
        default_conf.update(conf)
        extra_jars_packages = default_conf.get("spark.jars.packages")
        if extra_jars_packages:
            jars_packages.append(extra_jars_packages)
        extra_jars = default_conf.get("spark.jars")
        if extra_jars:
            jars.append(extra_jars)

    builder = SparkSession.builder.master(master_url).appName(
        "ScaleDP: v" + __version__,
    )

    for k, v in default_conf.items():
        builder.config(str(k), str(v))

    builder.config("spark.jars", ",".join(jars))
    builder.config("spark.jars.packages", ",".join(jars_packages))

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel(logLevel=logLevel)
    return spark


__all__ = [
    "ScaleDPSession",
    "DataToImage",
    "ImageDrawBoxes",
    "PdfDataToImage",
    "TesseractOcr",
    "Ner",
    "TextToDocument",
    "PipelineModel",
    "SuryaOcr",
    "EasyOcr",
    "DocTROcr",
    "YoloDetector",
    "ImageCropBoxes",
    "DSPyExtractor",
    "TesseractRecognizer",
    "DocTRTextDetector",
    "LLMVisualExtractor",
    "LLMExtractor",
    "LLMOcr",
    "LLMNer",
    "__version__",
    "files",
    *dir(enums),
]
