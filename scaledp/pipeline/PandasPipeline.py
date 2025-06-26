import itertools
import json
import logging
import time
from pathlib import Path
from typing import Any, ClassVar, List

import pandas as pd
from pyspark.ml import Transformer


class UserDefinedFunction:
    """
    User defined function in Python.

    .. versionadded:: 1.3

    Notes
    -----
    The constructor of this class is not supposed to be directly called.
    Use :meth:`pyspark.sql.functions.udf` or :meth:`pyspark.sql.functions.pandas_udf`
    to create this instance.

    """

    def __init__(
        self,
        func,
        returnType,
        name,
        evalType,
        deterministic: bool = True,
    ) -> None:
        if not callable(func):
            raise ValueError("Invalid function: not a function or callable")
        self.func = func
        self.returnType = returnType

    def __call__(self, *cols: Any) -> Any:
        cols = zip(*cols)
        return [self.func(*i) for i in cols]

    def _wrapped(self) -> Any:
        return self


def lit(value) -> Any:
    """Creates a :class:`Column` of literal value."""
    return itertools.repeat(value)


def _invoke_function(name: str, *args: Any) -> Any:
    if name == "lit":
        return lit(*args)
    raise ValueError("Invalid function name: %s" % name)


temp_functions = {}


def pathSparkFunctions(pyspark: Any) -> None:
    """Path Spark functions."""
    temp_functions["udf"] = pyspark.sql.udf.UserDefinedFunction
    pyspark.sql.udf.UserDefinedFunction = UserDefinedFunction
    temp_functions["invoke_function"] = pyspark.sql.functions._invoke_function
    pyspark.sql.functions._invoke_function = _invoke_function


def unpathSparkFunctions(pyspark: Any) -> None:
    """Unpath Spark functions."""
    pyspark.sql.udf.UserDefinedFunction = temp_functions["udf"]
    pyspark.sql.functions._invoke_function = temp_functions["invoke_function"]


class DatasetPd(pd.DataFrame):

    def withColumn(self, name, col) -> "DatasetPd":
        if name in self.columns:
            self[name] = col
        else:
            self.insert(0, name, col, True)
        return self

    def drop(self, col) -> "DatasetPd":
        return self

    def repartition(self, numPartitions) -> "DatasetPd":
        return self

    def coalesce(self, numPartitions) -> "DatasetPd":
        return self


class PandasPipeline:
    """PandasPipeline for call Spark ML Pipelines with Pandas DataFrame."""

    stages: ClassVar[List[Transformer]] = []

    def setStages(self, value) -> "PandasPipeline":
        self.stages = value
        return self

    def __init__(self, stages) -> None:
        self.setStages(stages)

    def fromFile(self, filename: str) -> Any:
        with Path.open(filename, "rb") as f:
            data = f.read()

        data = DatasetPd({"content": [data], "path": [filename], "resolution": [0]})

        return self.fromPandas(data)

    def fromBinary(self, data, filename="memory") -> Any:

        data = DatasetPd({"content": [data], "path": [filename], "resolution": [0]})
        return self.fromPandas(data)

    def fromPandas(self, data: pd.DataFrame) -> Any:

        start_time_total = time.time()
        execution_times = {"stages": []}
        data = DatasetPd(data)
        for stage in self.stages:

            stage_name = stage.__class__.__name__
            start_time_stage = time.time()
            data = stage._transform(data)

            stage_duration = time.time() - start_time_stage
            execution_times["stages"].append({stage_name: stage_duration})
            logging.info(
                f"Stage {stage_name} completed in {stage_duration:.2f} seconds",
            )

        total_duration = time.time() - start_time_total
        execution_times["total"] = total_duration
        logging.info(f"Total execution time: {total_duration:.2f} seconds")

        # Add execution time information as a new column
        return data.withColumn("execution_time", [json.dumps(execution_times)])

    def fromDict(self, data: dict) -> Any:
        data = DatasetPd(data)
        return self.fromPandas(data)
