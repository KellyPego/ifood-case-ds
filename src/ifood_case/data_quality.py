# src/ifood_case/data_quality.py

from typing import Dict, List, Optional, Union
from functools import reduce

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def get_total_rows(df: DataFrame) -> int:
    return df.count()


def null_summary(df: DataFrame, cols: Optional[List[str]] = None, total_rows: Optional[int] = None) -> DataFrame:
    """
    Null counts per column (+ total_rows).

    Args:
      cols: if provided, compute only for these columns
      total_rows: if provided, avoids an extra df.count()
    """
    if cols is None:
        cols = df.columns

    if total_rows is None:
        total_rows = df.count()

    exprs = [(F.sum(F.col(c).isNull().cast("int"))).alias(c) for c in cols]
    return df.select(exprs).withColumn("total_rows", F.lit(total_rows))


def key_check(df: DataFrame, key_cols: Union[str, List[str]], total_rows: Optional[int] = None) -> Dict[str, int]:
    """
    Checks:
      - total_rows
      - duplicate_key_groups (number of key combinations with count>1)
      - duplicate_rows (total duplicated rows beyond the first per key)
      - rows_with_null_keys

    Args:
      key_cols: column or list of columns that define the key
      total_rows: if provided, avoids an extra df.count()
    """
    if isinstance(key_cols, str):
        key_cols = [key_cols]

    if total_rows is None:
        total_rows = df.count()

    grouped = df.groupBy([F.col(c) for c in key_cols]).count()

    duplicate_key_groups = grouped.filter(F.col("count") > 1).count()
    duplicate_rows = grouped.filter(F.col("count") > 1).select((F.sum(F.col("count") - 1)).alias("dup")).collect()[0]["dup"]
    duplicate_rows = int(duplicate_rows) if duplicate_rows is not None else 0

    null_condition = reduce(lambda a, b: a | b, [F.col(c).isNull() for c in key_cols])
    rows_with_null_keys = df.filter(null_condition).count()

    return {
        "total_rows": int(total_rows),
        "duplicate_key_groups": int(duplicate_key_groups),
        "duplicate_rows": int(duplicate_rows),
        "rows_with_null_keys": int(rows_with_null_keys),
    }


def distinct_counts(
    df: DataFrame,
    cols: Union[str, List[str]],
    top_n: Optional[int] = 50,
    include_pct: bool = True,
    decimals: int = 4,
    total_rows: Optional[int] = None
) -> DataFrame:
    """
    Distinct values + count + (optional) percentage.

    Args:
      cols: column name or list of columns
      top_n: limit rows (None = no limit)
      include_pct: include percentage column
      decimals: decimal places for pct
      total_rows: if provided, avoids an extra df.count()
    """
    if isinstance(cols, str):
        cols = [cols]

    if total_rows is None:
        total_rows = df.count()

    out = df.groupBy([F.col(c) for c in cols]).count()

    if include_pct:
        out = out.withColumn("pct", F.round(F.col("count") / F.lit(total_rows), decimals))

    out = out.orderBy(F.col("count").desc())

    if top_n is not None:
        out = out.limit(int(top_n))

    return out