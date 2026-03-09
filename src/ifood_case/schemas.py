"""
Schemas for raw iFood case datasets.
"""

from pyspark.sql.types import (
    StructType, StructField, StringType, LongType, DoubleType, ArrayType
)

offers_schema = StructType([
    StructField("id", StringType(), True),
    StructField("offer_type", StringType(), True),
    StructField("min_value", LongType(), True),
    StructField("duration", DoubleType(), True),
    StructField("discount_value", LongType(), True),
    StructField("channels", ArrayType(StringType()), True)
])

profile_schema = StructType([
    StructField("age", LongType(), True),
    StructField("credit_card_limit", DoubleType(), True),
    StructField("gender", StringType(), True),
    StructField("id", StringType(), True),
    StructField("registered_on", StringType(), True)
])

transactions_schema = StructType([
    StructField("account_id", StringType(), True),
    StructField("event", StringType(), True),
    StructField("time_since_test_start", DoubleType(), True),
    StructField(
        "value",
        StructType([
            StructField("amount", DoubleType(), True),
            StructField("offer id", StringType(), True),
            StructField("offer_id", StringType(), True),
            StructField("reward", DoubleType(), True)
        ]),
        True
    )
])