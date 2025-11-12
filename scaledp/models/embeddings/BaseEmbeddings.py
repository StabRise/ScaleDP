import json

from pyspark.ml import Transformer
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql.functions import lit, pandas_udf, udf

from scaledp.params import (
    HasBatchSize,
    HasColumnValidator,
    HasDefaultEnum,
    HasDevice,
    HasInputCol,
    HasKeepInputData,
    HasModel,
    HasNumPartitions,
    HasOutputCol,
    HasPageCol,
    HasPartitionMap,
    HasPathCol,
)
from scaledp.schemas.EmbeddingsOutput import EmbeddingsOutput


class BaseEmbeddings(
    Transformer,
    HasInputCol,
    HasOutputCol,
    HasKeepInputData,
    HasDevice,
    HasModel,
    HasPathCol,
    DefaultParamsReadable,
    DefaultParamsWritable,
    HasNumPartitions,
    HasBatchSize,
    HasPageCol,
    HasColumnValidator,
    HasDefaultEnum,
    HasPartitionMap,
):

    def get_params(self):
        return json.dumps({k.name: v for k, v in self.extractParamMap().items()})

    def _transform(self, dataset):
        params = self.get_params()
        out_col = self.getOutputCol()
        input_col = self.getInputCol()
        if input_col not in dataset.columns:
            raise ValueError(f"Column {input_col} not found in dataset")
        in_col = self._validate(input_col, dataset)

        if not self.getPartitionMap():
            result = dataset.withColumn(
                out_col,
                udf(self.transform_udf, EmbeddingsOutput.get_schema())(in_col),
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
                pandas_udf(self.transform_udf_pandas, EmbeddingsOutput.get_schema())(
                    in_col,
                    lit(params),
                ),
            )

        if not self.getKeepInputData():
            result = result.drop(in_col)
        return result
