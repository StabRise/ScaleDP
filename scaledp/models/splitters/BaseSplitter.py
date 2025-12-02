from abc import ABC

from pyspark.ml import Transformer

from scaledp.params import (
    HasChunkOverlap,
    HasChunkSize,
    HasInputCol,
    HasKeepInputData,
    HasNumPartitions,
    HasOutputCol,
    HasPageCol,
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
    HasPageCol,
    ABC,
):
    """
    Abstract base class for text splitters.
    """
