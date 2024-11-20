
from pyspark.sql import DataFrame
from scaledp import DataToImage, PdfDataToImage, TesseractOcr, TextToDocument
import pyspark.sql.functions as f
import random
import base64

def _show_image(image, width=600, show_meta=True, index=0):
    from IPython.display import display, HTML
    from jinja2 import PackageLoader, Environment
    if image is None:
        print("Empty image")
        return

    img_base64 = base64.b64encode(image.data).decode("utf-8")

    templateEnv = Environment(loader=PackageLoader('scaledp.utils', 'templates'))
    template = templateEnv.get_template("image.html")
    metadata = {
        "Image#": index,
        "Path": image.path.split("/")[-1],
        "Size": f"{image.width} x {image.height} px",
        "Resolution": f"{image.resolution} dpi"
    }

    if image.exception != "":
        metadata["Exception"] = image.exception

    rendered_html = template.render(
        width=width,
        metadata=metadata,
        image=img_base64
    )

    display(HTML(rendered_html))

def get_column_type(df: DataFrame, column_name: str) -> str:
    for name, dtype in df.dtypes:
        if name == column_name:
            return dtype
    return None


def show_image(df, column="", limit=5, width=600, show_meta=True):
    if column == "":
        if "image" in df.columns:
            column = "image"
        elif "content" in df.columns:
            column = "content"
        else:
            raise ValueError("Please specify column name")
    column_type = get_column_type(df, column)
    if column_type == "binary":
        df = DataToImage().setInputCol(column).setOutputCol("image").transform(df)
        column = "image"
    for id_, row in enumerate(df.limit(limit).select(column).collect()):
        image = row[column]
        _show_image(image, width, show_meta, id_)


def show_text(df, column="", limit=5, width=800):
    from IPython.display import display, HTML
    from jinja2 import PackageLoader, Environment

    templateEnv = Environment(loader=PackageLoader('scaledp.utils', 'templates'))
    template = templateEnv.get_template("text.html")
    df = df.limit(limit)
    if column == "":
        if "value" in df.columns:
            column = "text"
            df = TextToDocument(inputCol="value").transform(df)
        elif "text" in df.columns:
            column = "text"
        elif "content" in df.columns:
            column = "text"
            df = DataToImage().transform(df)
            df = TesseractOcr(keepFormatting=True).transform(df)
        elif "image" in df.columns:
            column = "text"
            df = TesseractOcr(keepFormatting=True).transform(df)
        else:
            raise ValueError("Please specify column name")
    for id, text in enumerate(df.select(f"{column}.*").collect()):
        metadata = {
            "Id": id,
            "Path": text.path.split("/")[-1],
            "Exception": text.exception
        }

        rendered_html = template.render(
            width=width,
            metadata=metadata,
            text=text.text
        )

        display(HTML(rendered_html))


def show_pdf(df, column="", limit=5, width=600, show_meta=True):
    if column == "":
        if "pdf" in df.columns:
            column = "pdf"
        elif "content" in df.columns:
            column = "content"
        else:
            raise ValueError("Please specify column name")
    column_type = get_column_type(df, column)
    if column_type == "binary":
        df = PdfDataToImage(inputCol=column).transform(df)
        column = "image"
    else:
        raise ValueError("Column must be binary")
    for id_, row in enumerate(df.limit(limit).select(column).collect()):
        image = row[column]
        _show_image(image, width, show_meta, id_)


def show_ner(df, column="ner", limit=20, truncate=False):
    df.select(f.explode(f"{column}.entities").alias("ner")).select("ner.*").show(limit, truncate=truncate)


def visualize_ner(df, column="ner", text_column="text", limit=20, width=800, labels_list=None):
    from IPython.display import display, HTML
    from jinja2 import PackageLoader, Environment

    templateEnv = Environment(loader=PackageLoader('scaledp.utils', 'templates'))
    template = templateEnv.get_template("ner.html")

    df = df.limit(limit).select(column, text_column).cache()
    entities = df.select(f.explode(f"{column}.entities").alias("ner")).select("ner.*").collect()
    text = df.select(text_column).collect()[0][0]
    original_text = text.text

    entity_colors = {}
    current_position = 0
    html = ""
    for entity in entities:
        start = entity.start
        end = entity.end
        entity_name = entity.entity_group.lower()
        if entity_name not in entity_colors:
            entity_colors[entity_name] = "#{:06x}".format(random.randint(0, 0xFFFFFF))

        if current_position < start:
            html += "<span style='font-size:16px;line-height: 25px;'>" + original_text[
                                                                         current_position:start] + "</span>"
        if entity_name in entity_colors:
            html += f"""<span style='border-radius:4px;padding:2px;color:white;line-height:25px;margin:1px;
            background-color:{entity_colors[entity_name]};font-size:16px;white-space: nowrap;'>
            <span style="color:black;background-color:white;border-radius: 2px;padding: 0 1px 0 1px;">
            {original_text[start:end]}</span><span style='font-weight:500;padding: 4px;'>{entity.entity_group}</span></span>"""
        else:
            html += "<span style='font-size:16px;line-height: 25px;'>" + original_text[start:end] + "</span>"
        current_position = end

    metadata = {
        "Id": 0,
        "Path": text.path.split("/")[-1],
        "Exception": text.exception
    }

    rendered_html = template.render(
        width=width,
        metadata=metadata,
        ner=html
    )

    display(HTML(rendered_html))
