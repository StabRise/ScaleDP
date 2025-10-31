from scaledp.image.DataToImage import DataToImage


def test_data_to_image(raw_image_df):
    to_image = DataToImage()
    result = to_image.transform(raw_image_df).collect()

    assert len(result) == 1
    # present image field
    assert hasattr(result[0], "image")
    # image has right path field
    assert result[0].image.path == result[0].path
    assert result[0].image.exception == ""

    to_image = DataToImage()
    result = to_image.transform(raw_image_df)
    result1 = (
        DataToImage(inputCol="image.data", outputCol="image1")
        .transform(result)
        .collect()
    )

    assert len(result1) == 1
    # present image field
    assert hasattr(result1[0], "image")
    # image has right path field
    assert result1[0].image.path == result1[0].path
    assert result1[0].image.exception == ""


def test_data_to_image_without_path(raw_image_df):
    to_image = DataToImage()
    result = to_image.transform(raw_image_df.drop("path")).collect()

    assert len(result) == 1
    # present image field
    assert hasattr(result[0], "image")
    # image has right path field
    assert result[0].image.path == "memory"
    assert result[0].image.exception == ""


def test_wrong_data_to_image(binary_pdf_df):
    to_image = DataToImage()
    result = to_image.transform(binary_pdf_df).collect()

    assert len(result) == 1
    # present image field
    assert hasattr(result[0], "image")
    # has exception
    assert "Unable to read image" in result[0].image.exception
