from pathlib import Path

import pyspark
import pytest
from PIL import Image as pImage

from scaledp.enums import ImageType
from scaledp.image.DataToImage import DataToImage
from scaledp.pipeline.PandasPipeline import pathSparkFunctions, unpathSparkFunctions
from scaledp.schemas.Image import Image
from scaledp.schemas.TextChunks import TextChunks


@pytest.fixture
def patch_spark():
    """Fixture to handle patching and unpatching of spark functions."""
    pathSparkFunctions(pyspark)
    yield
    unpathSparkFunctions(pyspark)


@pytest.fixture
def image_file(resource_path_root):
    return (resource_path_root / "images/Invoice.png").absolute().as_posix()


@pytest.fixture
def image_rotated_text_file(resource_path_root):
    return (resource_path_root / "images/RotatedText.png").absolute().as_posix()


@pytest.fixture
def image_rotated_text_df(spark_session, image_rotated_text_file):
    df = spark_session.read.format("binaryFile").load(
        image_rotated_text_file,
    )
    bin_to_image = DataToImage().setImageType(ImageType.WEBP.value)
    return bin_to_image.transform(df)


@pytest.fixture
def receipt_file(resource_path_root):
    return (resource_path_root / "images" / "receipt.jpg").absolute().as_posix()


@pytest.fixture
def image_pil(image_file):
    return pImage.open(image_file)


@pytest.fixture
def image_pil_1x1() -> pImage.Image:
    return pImage.new("RGB", (1, 1), color="red")


@pytest.fixture
def image(image_pil: pImage.Image) -> Image:

    return Image.from_pil(image_pil, "test", ImageType.FILE.value, 300)


@pytest.fixture
def image_line(resource_path_root):
    from scaledp.schemas.Image import Image

    return Image.from_pil(
        pImage.open(
            (resource_path_root / "images/text_line.png").absolute().as_posix(),
        ),
        "test",
        ImageType.FILE.value,
        300,
    )


@pytest.fixture
def raw_image_df(spark_session, resource_path_root):
    return spark_session.read.format("binaryFile").load(
        (resource_path_root / "images/Invoice.png").absolute().as_posix(),
    )


@pytest.fixture
def binary_pdf_df(spark_session, resource_path_root):
    return spark_session.read.format("binaryFile").load(
        (resource_path_root / "pdfs/unipdf-medical-bill.pdf").absolute().as_posix(),
    )


@pytest.fixture
def pdf_df(spark_session, resource_path_root):
    return spark_session.read.format("binaryFile").load(
        (resource_path_root / "pdfs/unipdf-medical-bill.pdf").absolute().as_posix(),
    )


@pytest.fixture
def pdf_df_extra(spark_session, resource_path_root):
    return spark_session.read.format("binaryFile").load(
        (resource_path_root / "pdfs/Example3.pdf").absolute().as_posix(),
    )


@pytest.fixture
def pdf_horizontal_df(spark_session, resource_path_root):
    return spark_session.read.format("binaryFile").load(
        (resource_path_root / "pdfs/horizontal.pdf").absolute().as_posix(),
    )


@pytest.fixture
def image_pdf_df(spark_session, resource_path_root):
    return spark_session.read.format("binaryFile").load(
        (resource_path_root / "pdfs/image_pdf.pdf").absolute().as_posix(),
    )


@pytest.fixture
def signatures_pdf_file(resource_path_root):
    return (
        (resource_path_root / "pdfs" / "SampleWithSignatures.pdf").absolute().as_posix()
    )


@pytest.fixture
def signatures_pdf_df(spark_session, signatures_pdf_file):
    return spark_session.read.format("binaryFile").load(
        signatures_pdf_file,
    )


@pytest.fixture
def face_pdf_file(spark_session, resource_path_root):
    return (resource_path_root / "pdfs" / "SampleWithFace.pdf").absolute().as_posix()


@pytest.fixture
def pdf_file(resource_path_root):
    return (resource_path_root / "pdfs/unipdf-medical-bill.pdf").absolute().as_posix()


@pytest.fixture
def pdf_report_file(resource_path_root):
    return (resource_path_root / "pdfs/sample-report.pdf").absolute().as_posix()


@pytest.fixture
def image_df(spark_session, resource_path_root):
    df = spark_session.read.format("binaryFile").load(
        (resource_path_root / "images/Invoice.png").absolute().as_posix(),
    )
    bin_to_image = DataToImage().setImageType(ImageType.WEBP.value)
    return bin_to_image.transform(df)


@pytest.fixture
def image_line_df(spark_session, resource_path_root):
    df = spark_session.read.format("binaryFile").load(
        (resource_path_root / "images/text_line.png").absolute().as_posix(),
    )
    bin_to_image = DataToImage().setImageType(ImageType.WEBP.value)
    return bin_to_image.transform(df)


@pytest.fixture
def image_receipt_df(spark_session, resource_path_root):
    df = spark_session.read.format("binaryFile").load(
        (resource_path_root / "images" / "receipt.jpg").absolute().as_posix(),
    )
    bin_to_image = DataToImage().setImageType(ImageType.WEBP.value)
    return bin_to_image.transform(df)


@pytest.fixture
def image_rotated_df(spark_session, resource_path_root):
    df = spark_session.read.format("binaryFile").load(
        (resource_path_root / "images" / "img_rotated.png").absolute().as_posix(),
    )
    bin_to_image = DataToImage().setImageType(ImageType.WEBP.value)
    return bin_to_image.transform(df)


@pytest.fixture
def image_qr_code_df(spark_session, resource_path_root):
    df = spark_session.read.format("binaryFile").load(
        (resource_path_root / "images" / "QrCode.png").absolute().as_posix(),
    )
    bin_to_image = DataToImage().setImageType(ImageType.WEBP.value)
    return bin_to_image.transform(df)


@pytest.fixture
def image_signature_df(spark_session, resource_path_root):
    df = spark_session.read.format("binaryFile").load(
        (resource_path_root / "images" / "signature.png").absolute().as_posix(),
    )
    bin_to_image = DataToImage().setImageType(ImageType.WEBP.value)
    return bin_to_image.transform(df)


@pytest.fixture
def image_face_df(spark_session, resource_path_root):
    df = spark_session.read.format("binaryFile").load(
        (resource_path_root / "images" / "document_with_face.png")
        .absolute()
        .as_posix(),
    )
    bin_to_image = DataToImage().setImageType(ImageType.WEBP.value)
    return bin_to_image.transform(df)


@pytest.fixture
def receipt_json(receipt_json_path: Path) -> Path:
    return receipt_json_path.open("r").read()


@pytest.fixture
def receipt_json_path(resource_path_root: Path) -> Path:
    return resource_path_root / "images" / "receipt.json"


@pytest.fixture
def receipt_with_null_json(receipt_json_path: Path) -> Path:
    return receipt_json_path.open("r").read()


@pytest.fixture
def receipt_with_null_json_path(resource_path_root: Path) -> Path:
    return resource_path_root / "images" / "receipt_with_null.json"


@pytest.fixture
def text_df(spark_session, resource_path_root):
    return spark_session.read.text(
        (resource_path_root / "texts/example.txt").absolute().as_posix(),
        wholetext=True,
    )


@pytest.fixture
def document_df(spark_session, resource_path_root):
    """Fixture for text splitter tests with Document struct column."""
    from pyspark.sql.functions import col, lit, struct
    from pyspark.sql.types import ArrayType, StructType

    text_path = (resource_path_root / "texts/example.txt").absolute().as_posix()
    df = spark_session.read.text(text_path, wholetext=True)

    # Create Document struct from text and path
    return (
        df.withColumn(
            "document",
            struct(
                lit(text_path).alias("path"),
                col("value").alias("text"),
                lit("text").alias("type"),
                lit([]).cast(ArrayType(StructType([]))).alias("bboxes"),
                lit("").alias("exception"),
            ),
        )
        .withColumn("page_number", lit(0))
        .select("document", "page_number")
    )


@pytest.fixture
def df_text_chunks(spark_session):
    """Fixture for TextChunks schema with sample data."""
    from pyspark.sql.functions import struct

    chunks_data = [
        ("file1.txt", 0, ["hello world", "this is a test"], "", 1.0),
        ("file2.txt", 0, ["another chunk", "more text"], "", 2.0),
        ("file3.txt", 0, ["final chunk"], "", 0.5),
    ]

    return spark_session.createDataFrame(
        chunks_data,
        schema=TextChunks.get_schema(),
    ).select(
        struct("path", "page", "chunks", "exception", "processing_time").alias(
            "text_chunks_col",
        ),
    )


@pytest.fixture
def extraction_conditions_pdf_file(resource_path_root):
    """Rendering condition matrix page from pdf-redaction-benchmarks v0.1.1
    (case extraction-conditions-1): the same name rendered under many
    orientations, polarities, provenances and sizes.
    """
    return (
        (resource_path_root / "pdfs" / "ExtractionConditions.pdf").absolute().as_posix()
    )


@pytest.fixture
def extraction_conditions_rotated_probes():
    """Ground truth for the rotated/turned/stacked cells of
    ExtractionConditions.pdf. Every cell contains "Freya Yamamoto".

    Channel is "text_layer" (real text in the PDF) or "image_text" (rasterised,
    no text layer). Bboxes are converted from PDF user space (bottom-left
    origin) to top-left origin, in points.
    """
    page_height = 841.89
    probes = [
        ("t032", "text_layer", "rot180", (54.00, 340.27, 138.00, 349.27)),
        ("t033", "text_layer", "skew", (153.86, 333.40, 237.32, 363.83)),
        ("t034", "text_layer", "rot180", (253.71, 340.27, 337.71, 349.27)),
        ("t035", "image_text", "skew", (353.57, 326.86, 444.17, 363.83)),
        ("t036", "image_text", "skew", (453.42, 331.83, 522.25, 363.83)),
        ("t037", "image_text", "rot180", (54.00, 282.53, 144.00, 296.69)),
        ("t038", "text_layer", "skew", (153.86, 281.37, 203.94, 299.63)),
        ("t039", "image_text", "skew", (253.71, 281.03, 307.18, 303.11)),
        ("t040", "image_text", "rot180", (383.57, 282.53, 450.77, 297.65)),
        ("t041", "text_layer", "rot90", (54.50, 158.35, 63.50, 242.35)),
        ("t042", "text_layer", "rot270", (90.02, 158.35, 99.02, 242.35)),
        ("t043", "text_layer", "vertical", (125.04, 96.85, 131.04, 242.35)),
        ("t044", "text_layer", "rot90", (161.06, 158.35, 170.06, 242.35)),
        ("t045", "text_layer", "vertical", (196.08, 96.85, 202.08, 242.35)),
        ("t046", "image_text", "rot270", (231.60, 152.35, 245.76, 242.35)),
        ("t047", "image_text", "rot90", (267.12, 179.47, 281.28, 242.35)),
        ("t048", "text_layer", "rot270", (303.14, 158.35, 312.14, 242.35)),
        ("t049", "image_text", "rot90", (338.16, 152.35, 352.32, 242.35)),
        ("t050", "image_text", "rot270", (373.68, 179.47, 387.84, 242.35)),
        ("t051", "text_layer", "vertical", (409.20, 155.05, 412.80, 242.35)),
        ("t052", "text_layer", "rot90", (445.02, 191.95, 450.42, 242.35)),
        ("t053", "text_layer", "rot90", (480.54, 191.95, 485.94, 242.35)),
        ("t054", "text_layer", "vertical", (515.76, 155.05, 519.36, 242.35)),
    ]
    return [
        {
            "id": probe_id,
            "channel": channel,
            "orientation": orientation,
            "bbox": (x0, page_height - y1, x1, page_height - y0),
        }
        for probe_id, channel, orientation, (x0, y0, x1, y1) in probes
    ]
