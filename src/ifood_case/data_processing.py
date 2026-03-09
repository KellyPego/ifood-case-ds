"""
Data processing module.

Provides functions to clean, restructure, and join the raw iFood case study
DataFrames into a unified dataset.

Important:
- The raw "transactions" dataset is event-based (offer received/viewed/completed/transaction).
- df_joined is an enriched event log (one row per event).
- For EDA Q1-Q5 and modeling, use opps_enriched:
    one row per "offer received" with flags viewed/completed inside the offer window.

Public API:
    process_data(offers, transactions, profile) -> (df_joined, offers, transactions_processed, profile_processed)
    build_opps_enriched(df_joined) -> DataFrame
"""

from typing import Tuple

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def process_data(
    offers: DataFrame,
    transactions: DataFrame,
    profile: DataFrame,
) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    """
    Clean and join raw DataFrames into a unified event log.

    Returns:
        df_joined: enriched event log (one row per event)
        offers: offers DataFrame (unchanged)
        transactions_processed: flattened transactions DataFrame
        profile_processed: profile with age_group and cleaned gender
    """
    transactions_processed = _restructure_transactions(transactions)
    profile_processed = _create_age_groups_and_clean_gender(profile)
    df_joined = _join_dataframes(
        transactions=transactions_processed,
        profile=profile_processed,
        offers=offers,
    )
    return df_joined, offers, transactions_processed, profile_processed


def build_opps_enriched(df_joined: DataFrame) -> DataFrame:
    """
    Build opportunities dataset from df_joined.

    Output grain:
      - one row per (user_id, offer_id, t_received)

    Adds:
      - viewed:    1 if offer viewed inside window
      - completed: 1 if offer completed inside window
      - limit_band: quartiles from credit_card_limit

    This is the recommended dataset for:
      - EDA questions Q1-Q5
      - ABT / modeling (after feature engineering)
    """
    opps = _build_opportunities(df_joined)
    opps = _add_limit_band(opps)
    return opps


# ── private helpers ────────────────────────────────────────────────────────────

def _restructure_transactions(transactions_df: DataFrame) -> DataFrame:
    """
    Flatten nested 'value' fields in transactions.

    The dataset has offer id in two possible nested fields:
    - value.offer_id
    - value.`offer id` (with space)

    We unify both into a single column: offer_id.

    IMPORTANT:
    - We MUST drop the struct column 'value' afterward, otherwise Delta will fail
        when saving due to invalid nested field names (e.g., 'offer id').
    """
    if "value" not in transactions_df.columns:
        return transactions_df

    return (
        transactions_df
        .withColumn("offer_id", F.coalesce(F.col("value.offer_id"), F.col("value.`offer id`")))
        .withColumn("amount", F.col("value.amount"))
        .withColumn("reward", F.col("value.reward"))
        .drop("value")
    )


def _create_age_groups_and_clean_gender(profile_df: DataFrame) -> DataFrame:
    """
    Add age_group and clean gender.

    Rules:
      - gender NULL/empty/unknown -> "Unknown"
      - valid gender: F, M, O

    Fix:
      - Treat age == 118 as missing (NULL) before creating age_group.
    """
    df = profile_df

    if "age" in df.columns:
        df = df.withColumn("age", F.col("age").cast("int"))
        df = df.withColumn(
            "age",
            F.when(F.col("age") == 118, F.lit(None).cast("int"))
             .otherwise(F.col("age"))
        )
        df = df.withColumn(
            "age_group",
            F.when((F.col("age") >= 18) & (F.col("age") <= 24), "18-24")
             .when((F.col("age") >= 25) & (F.col("age") <= 34), "25-34")
             .when((F.col("age") >= 35) & (F.col("age") <= 44), "35-44")
             .when((F.col("age") >= 45) & (F.col("age") <= 54), "45-54")
             .when((F.col("age") >= 55) & (F.col("age") <= 64), "55-64")
             .when(F.col("age") > 64, "65+")
             .otherwise("Unknown"),
        )

    if "gender" in df.columns:
        df = (
            df.withColumn("gender", F.upper(F.trim(F.col("gender"))))
              .withColumn(
                  "gender",
                  F.when(
                      (F.col("gender").isNull()) | (F.col("gender") == "") | (~F.col("gender").isin("F", "M", "O")),
                      F.lit("Unknown")
                  ).otherwise(F.col("gender"))
              )
        )

    return df


def _join_dataframes(
    transactions: DataFrame,
    profile: DataFrame,
    offers: DataFrame,
) -> DataFrame:
    """
    Join transactions + profile + offers using the real keys available in the dataset.

    Keys:
      - transactions.account_id = profile.id
      - transactions.offer_id   = offers.id
    """
    t = transactions.alias("t")
    p = profile.alias("p")
    o = offers.alias("o")

    return (
        t
        .join(p, F.col("t.account_id") == F.col("p.id"), "left")
        .join(o, F.col("t.offer_id") == F.col("o.id"), "left")
        .select(
            F.col("t.account_id").alias("user_id"),
            F.col("t.event").alias("event"),
            F.col("t.time_since_test_start").cast("double").alias("time_since_test_start"),
            F.col("t.offer_id").alias("offer_id"),
            F.col("t.amount").alias("amount"),
            F.col("t.reward").alias("reward"),
            F.col("p.age").alias("age"),
            F.col("p.gender").alias("gender"),
            F.col("p.credit_card_limit").alias("credit_card_limit"),
            F.col("p.age_group").alias("age_group"),
            F.col("o.offer_type").alias("offer_type"),
            F.col("o.min_value").alias("min_value"),
            F.col("o.duration").cast("double").alias("duration"),
            F.col("o.discount_value").alias("discount_value"),
            F.col("o.channels").alias("channels"),
            F.to_date(F.col("p.registered_on"), "yyyyMMdd").alias("registered_on_date"),
        )
    )


def _build_opportunities(df_joined: DataFrame) -> DataFrame:
    """
    Build opportunity table (one row per received offer).

    Grain:
      - (user_id, offer_id, t_received)

    Flags:
      - viewed:    1 if offer viewed within [t_received, t_received + duration]
      - completed: 1 if offer completed within [t_received, t_received + duration]

    IMPORTANT FIX:
      - Do NOT filter rows after left join (it can drop opportunities).
      - Instead, conditionally null out out-of-window events and aggregate with min().
    """
    base = df_joined.select(
        "user_id",
        "event",
        F.col("time_since_test_start").cast("double").alias("t"),
        "offer_id",
        "offer_type",
        F.col("duration").cast("double").alias("duration"),
        "age",
        "age_group",
        "gender",
        "credit_card_limit",
        "min_value",
        "discount_value",
        "channels",
    )

    received = (
        base
        .filter(F.col("event") == "offer received")
        .select(
            "user_id",
            "offer_id",
            "offer_type",
            "duration",
            "age",
            "age_group",
            "gender",
            "credit_card_limit",
            "min_value",
            "discount_value",
            "channels",
            F.col("t").alias("t_received"),
        )
        .withColumn("t_expires", F.col("t_received") + F.col("duration"))
    )

    viewed = (
        base
        .filter(F.col("event") == "offer viewed")
        .select("user_id", "offer_id", F.col("t").alias("t_viewed"))
    )

    completed = (
        base
        .filter(F.col("event") == "offer completed")
        .select("user_id", "offer_id", F.col("t").alias("t_completed"))
    )

    received_view = (
        received
        .join(viewed, on=["user_id", "offer_id"], how="left")
        .withColumn(
            "t_viewed_in_window",
            F.when(
                (F.col("t_viewed") >= F.col("t_received")) & (F.col("t_viewed") <= F.col("t_expires")),
                F.col("t_viewed")
            )
        )
        .groupBy(*received.columns)
        .agg(F.min("t_viewed_in_window").alias("t_viewed"))
    )

    opps = (
        received_view
        .join(completed, on=["user_id", "offer_id"], how="left")
        .withColumn(
            "t_completed_in_window",
            F.when(
                (F.col("t_completed") >= F.col("t_received")) & (F.col("t_completed") <= F.col("t_expires")),
                F.col("t_completed")
            )
        )
        .groupBy(*received_view.columns)
        .agg(F.min("t_completed_in_window").alias("t_completed"))
        .withColumn("viewed", F.when(F.col("t_viewed").isNotNull(), F.lit(1)).otherwise(F.lit(0)))
        .withColumn("completed", F.when(F.col("t_completed").isNotNull(), F.lit(1)).otherwise(F.lit(0)))
    )

    return opps


def _add_limit_band(opps: DataFrame) -> DataFrame:
    """
    Add credit_card_limit quartile bands as limit_band.
    Serverless-safe (no RDD usage).
    """
    credit_base = opps.select("credit_card_limit").where(F.col("credit_card_limit").isNotNull())

    has_any = credit_base.limit(1).count() > 0
    if not has_any:
        return opps.withColumn("limit_band", F.lit("Unknown"))

    q1, q2, q3 = credit_base.approxQuantile("credit_card_limit", [0.25, 0.50, 0.75], 0.01)

    return (
        opps.withColumn(
            "limit_band",
            F.when(F.col("credit_card_limit").isNull(), F.lit("Unknown"))
            .when(F.col("credit_card_limit") <= F.lit(q1), F.lit(f"<= {int(q1)}"))
            .when(F.col("credit_card_limit") <= F.lit(q2), F.lit(f"{int(q1)} - {int(q2)}"))
            .when(F.col("credit_card_limit") <= F.lit(q3), F.lit(f"{int(q2)} - {int(q3)}"))
            .otherwise(F.lit(f"> {int(q3)}"))
        )
    )
