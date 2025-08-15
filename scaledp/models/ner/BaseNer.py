import json

from pyspark.ml import Transformer
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql.functions import lit, pandas_udf, udf
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from scaledp.params import (
    HasBatchSize,
    HasColumnValidator,
    HasDefaultEnum,
    HasDevice,
    HasInputCols,
    HasKeepInputData,
    HasModel,
    HasNumPartitions,
    HasOutputCol,
    HasPageCol,
    HasPartitionMap,
    HasPathCol,
    HasScoreThreshold,
    HasWhiteList,
)
from scaledp.schemas.NerOutput import NerOutput


class BaseNer(
    Transformer,
    HasInputCols,
    HasOutputCol,
    HasKeepInputData,
    HasWhiteList,
    HasDevice,
    HasModel,
    HasPathCol,
    DefaultParamsReadable,
    DefaultParamsWritable,
    HasNumPartitions,
    HasScoreThreshold,
    HasBatchSize,
    HasPageCol,
    HasColumnValidator,
    HasDefaultEnum,
    HasPartitionMap,
):

    def outputSchema(self):
        return StructType(
            [
                StructField("path", StringType(), True),
                StructField(
                    "entities",
                    ArrayType(
                        StructType(
                            [
                                StructField("entity_group", StringType(), False),
                                StructField("score", DoubleType(), False),
                                StructField("word", StringType(), False),
                                StructField("start", IntegerType(), False),
                                StructField("end", IntegerType(), False),
                                StructField(
                                    "boxes",
                                    ArrayType(
                                        StructType(
                                            [
                                                StructField(
                                                    "text",
                                                    StringType(),
                                                    False,
                                                ),
                                                StructField(
                                                    "score",
                                                    DoubleType(),
                                                    False,
                                                ),
                                                StructField("x", IntegerType(), False),
                                                StructField("y", IntegerType(), False),
                                                StructField(
                                                    "width",
                                                    IntegerType(),
                                                    False,
                                                ),
                                                StructField(
                                                    "height",
                                                    IntegerType(),
                                                    False,
                                                ),
                                            ],
                                        ),
                                        True,
                                    ),
                                    False,
                                ),
                            ],
                        ),
                        True,
                    ),
                    True,
                ),
                StructField("exception", StringType(), True),
            ],
        )

    def get_params(self):
        return json.dumps({k.name: v for k, v in self.extractParamMap().items()})

    def _transform(self, dataset):
        params = self.get_params()
        out_col = self.getOutputCol()
        for col in self.getInputCols():
            if col not in dataset.columns:
                raise ValueError(f"Column {col} not found in dataset")
        in_cols = [self._validate(col, dataset) for col in self.getInputCols()]

        if not self.getPartitionMap():
            result = dataset.withColumn(
                out_col,
                udf(self.transform_udf, NerOutput.get_schema())(*in_cols),
            )
        else:
            if self.getNumPartitions() > 0:
                if self.getPageCol() in dataset.columns:
                    dataset = dataset.repartition(self.getPageCol())
                elif self.getPathCol() in dataset.columns:
                    dataset = dataset.repartition(self.getPathCol())
                dataset = dataset.coalesce(self.getNumPartitions())
            result = dataset.withColumn(
                out_col,
                pandas_udf(self.transform_udf_pandas, self.outputSchema())(
                    *in_cols,
                    lit(params),
                ),
            )

        if not self.getKeepInputData():
            result = result.drop(*in_cols)
        return result
