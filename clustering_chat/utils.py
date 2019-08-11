import math
from typing import List, Union

from numpy import array_split
from pandas import DataFrame

ArrayLike = Union[list, DataFrame]


def partition_df(df: DataFrame, n: int) -> List[DataFrame]:
    sections = math.ceil(len(df.index) / n)
    split = array_split(df, sections)
    return split


def flatten(lists: List[list]) -> list:
    return sum(lists, [])
