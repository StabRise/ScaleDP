from pathlib import Path

import pytest
from PIL import Image as pImage

from scaledp.enums import ImageType
from scaledp.image.DataToImage import DataToImage
from scaledp.schemas.Image import Image


@pytest.fixture
def image_file(resource_path_root):
    return (resource_path_root / "images/Invoice.png").absolute().as_posix()


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
def image_pdf_df(spark_session, resource_path_root):
    return spark_session.read.format("binaryFile").load(
        (resource_path_root / "pdfs/image_pdf.pdf").absolute().as_posix(),
    )


@pytest.fixture
def pdf_file(resource_path_root):
    return (resource_path_root / "pdfs/unipdf-medical-bill.pdf").absolute().as_posix()


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
