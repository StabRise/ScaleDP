from abc import ABC

from pyspark.ml import Transformer

from scaledp.params import (
    HasChunkOverlap,
    HasChunkSize,
    HasInputCol,
    HasKeepInputData,
    HasNumPartitions,
    HasOutputCol,
    HasPartitionMap,
    HasWhiteList,
)


class BaseSplitter(
    Transformer,
    HasInputCol,
    HasOutputCol,
    HasKeepInputData,
    HasWhiteList,
    HasNumPartitions,
    HasPartitionMap,
    HasChunkSize,
    HasChunkOverlap,
    ABC,
):
    """
    Abstract base class for text splitters.
    """
