"""
Feature engineering for the iFood propensity case.

Goal:
  - Build an opportunity-level ABT (one row per offer received)
  - Strict point-in-time features (no leakage)
  - Target:
      * bogo/discount: completed within offer window
      * informational: any transaction within offer window
  - Clean, readable feature names (production-friendly)
  - Avoid redundant features (do not create what we plan to drop)

Lookbacks:
  - short: 7 days
  - long:  28 days

Public API:
    build_abt(df_joined, opps_enriched, lookback=LookbackConfig(), known_channels=None)
      -> (abt_df, numerical_cols, categorical_cols)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window


@dataclass(frozen=True)
class LookbackConfig:
    days_short: float = 7.0
    days_long: float = 28.0


def build_abt(
    df_joined: DataFrame,
    opps_enriched: DataFrame,
    lookback: LookbackConfig = LookbackConfig(),
    known_channels: Optional[List[str]] = None,
) -> Tuple[DataFrame, List[str], List[str]]:
    """
    Build the opportunity-level ABT for propensity modeling.

    Args:
        df_joined:      enriched event log (one row per event)
        opps_enriched:  one row per received offer with window boundaries and flags
        lookback:       lookback window config (days_short, days_long)
        known_channels: list of channel names to create binary flags for

    Returns:
        abt_df:          modeling table with 'target' + point-in-time features
        numerical_cols:  list of numerical feature column names
        categorical_cols: list of categorical feature column names
    """
    channels = known_channels or ["web", "email", "mobile", "social"]

    base = _base_opportunities(opps_enriched)
    base_with_target = _build_target(base, df_joined)

    offer_feats = _offer_static_features(base_with_target)
    channel_feats = _channel_features(base_with_target, channels)
    history_feats = _historical_features(base_with_target, df_joined, lookback)
    last_offer_feats = _last_offer_viewed_type(base_with_target, df_joined)
    tenure_feats = _customer_tenure_feature(base_with_target, df_joined)

    abt_df = (
        base_with_target.alias("b")
        .join(offer_feats.alias("o"), on=["customer_id", "offer_id", "time_received"], how="left")
        .join(channel_feats.alias("c"), on=["customer_id", "offer_id", "time_received"], how="left")
        .join(history_feats.alias("h"), on=["customer_id", "offer_id", "time_received"], how="left")
        .join(last_offer_feats.alias("l"), on=["customer_id", "offer_id", "time_received"], how="left")
        .join(tenure_feats.alias("ten"), on=["customer_id", "offer_id", "time_received"], how="left")
    )

    abt_df = abt_df.withColumn(
        "reward_x_avg_ticket",
        F.col("offer_reward_ratio") * F.col("avg_ticket_28d"),
    )

    abt_df = _final_null_handling(abt_df)
    numerical_cols, categorical_cols = _infer_column_types(abt_df)

    return abt_df, numerical_cols, categorical_cols


# ── private helpers ────────────────────────────────────────────────────────────

def _base_opportunities(opps: DataFrame) -> DataFrame:
    required = ["user_id", "offer_id", "t_received", "t_expires", "offer_type", "duration", "completed", "viewed"]
    missing = [c for c in required if c not in opps.columns]
    if missing:
        raise ValueError(f"opps_enriched is missing required columns: {missing}")

    base = (
        opps.select(
            F.col("user_id").alias("customer_id"),
            F.col("offer_id").alias("offer_id"),
            F.col("t_received").cast("double").alias("time_received"),
            F.col("t_expires").cast("double").alias("time_expires"),
            F.col("offer_type").cast("string").alias("offer_type"),
            F.col("duration").cast("double").alias("offer_duration_days"),
            F.col("min_value").alias("offer_min_spend"),
            F.col("discount_value").alias("offer_discount_value"),
            F.col("channels").alias("offer_channels"),
            F.col("viewed").cast("int").alias("offer_viewed_in_window"),
            F.col("completed").cast("int").alias("offer_completed_in_window"),
            (F.col("age").cast("double") if "age" in opps.columns else F.lit(None).cast("double")).alias("customer_age"),
            (F.col("gender").cast("string") if "gender" in opps.columns else F.lit("Unknown").cast("string")).alias("customer_gender"),
            (F.col("credit_card_limit").cast("double") if "credit_card_limit" in opps.columns else F.lit(None).cast("double")).alias("customer_credit_limit"),
        )
    )

    dup_groups = (
        base.groupBy("customer_id", "offer_id", "time_received")
        .count()
        .filter(F.col("count") > 1)
        .count()
    )
    if dup_groups > 0:
        raise ValueError(
            f"opps_enriched grain is not unique: found {dup_groups} duplicate key groups "
            f"for (customer_id, offer_id, time_received)"
        )

    return base


def _build_target(base: DataFrame, df_joined: DataFrame) -> DataFrame:
    """
    Target definition:
      - bogo/discount:    target = offer_completed_in_window
      - informational:    target = 1 if ANY transaction occurs within [time_received, time_expires]
    """
    tx = (
        df_joined
        .filter(F.col("event") == "transaction")
        .select(
            F.col("user_id").alias("customer_id"),
            F.col("time_since_test_start").cast("double").alias("t_tx"),
        )
    )

    info_tx = (
        base
        .filter(F.col("offer_type") == "informational")
        .select("customer_id", "offer_id", "time_received", "time_expires")
        .join(tx, on="customer_id", how="left")
        .where((F.col("t_tx") >= F.col("time_received")) & (F.col("t_tx") <= F.col("time_expires")))
        .groupBy("customer_id", "offer_id", "time_received")
        .agg(F.lit(1).alias("_target_info"))
    )

    return (
        base
        .join(info_tx, on=["customer_id", "offer_id", "time_received"], how="left")
        .withColumn("_target_info", F.coalesce(F.col("_target_info"), F.lit(0)))
        .withColumn(
            "target",
            F.when(F.col("offer_type") == "informational", F.col("_target_info"))
             .otherwise(F.col("offer_completed_in_window"))
             .cast("int")
        )
        .drop("_target_info", "offer_viewed_in_window", "offer_completed_in_window")
    )


def _offer_static_features(base: DataFrame) -> DataFrame:
    df = base.select(
        "customer_id",
        "offer_id",
        "time_received",
        F.col("offer_duration_days").cast("double").alias("offer_duration_days"),
        F.col("offer_min_spend").cast("double").alias("offer_min_spend"),
        F.col("offer_discount_value").cast("double").alias("offer_discount_value"),
    )

    df = df.withColumn(
        "offer_reward_ratio",
        F.when(
            F.col("offer_min_spend").isNotNull()
            & (F.col("offer_min_spend") > 0)
            & F.col("offer_discount_value").isNotNull(),
            F.col("offer_discount_value") / F.col("offer_min_spend"),
        ).otherwise(F.lit(0.0))
    )

    df = df.withColumn(
        "offer_duration_bucket",
        F.when(F.col("offer_duration_days") <= 3, F.lit("short"))
         .when(F.col("offer_duration_days") <= 7, F.lit("medium"))
         .otherwise(F.lit("long"))
    )

    return df.select(
        "customer_id", "offer_id", "time_received",
        "offer_reward_ratio", "offer_duration_bucket"
    )


def _channel_features(base: DataFrame, known_channels: List[str]) -> DataFrame:
    if "offer_channels" not in base.columns:
        out = base.select("customer_id", "offer_id", "time_received")
        for ch in known_channels:
            out = out.withColumn(f"channel_is_{ch}", F.lit(0).cast("int"))
        return out.withColumn("offer_channel_count", F.lit(0).cast("int"))

    df = base.select("customer_id", "offer_id", "time_received", "offer_channels")

    for ch in known_channels:
        df = df.withColumn(
            f"channel_is_{ch}",
            F.when(F.array_contains(F.col("offer_channels"), ch), F.lit(1)).otherwise(F.lit(0)).cast("int")
        )

    df = df.withColumn("offer_channel_count", F.size(F.col("offer_channels")).cast("int"))
    return df.drop("offer_channels")


def _historical_features(base: DataFrame, df_joined: DataFrame, lookback: LookbackConfig) -> DataFrame:
    """
    Point-in-time features using df_joined event log.

    We KEEP (non-redundant):
      - tx_count_7d, tx_count_28d, tx_amount_sum_28d, days_since_last_tx
      - offers_viewed_count_28d, offers_completed_count_28d
      - offer_view_rate_28d, offer_completion_rate_28d
      - days_since_last_offer_view, days_since_last_offer_completion
    """
    t = base.select("customer_id", "offer_id", "time_received")

    events = (
        df_joined.select(
            F.col("user_id").alias("customer_id"),
            F.col("event").alias("event"),
            F.col("time_since_test_start").cast("double").alias("t_event"),
            (F.col("amount").cast("double") if "amount" in df_joined.columns else F.lit(None).cast("double")).alias("amount"),
        )
    )

    hist = (
        t.join(events, on="customer_id", how="left")
         .where(F.col("t_event") < F.col("time_received"))
    )

    in_short = F.col("t_event") >= (F.col("time_received") - F.lit(lookback.days_short))
    in_long = F.col("t_event") >= (F.col("time_received") - F.lit(lookback.days_long))

    tx_agg = (
        hist.where(F.col("event") == "transaction")
            .groupBy("customer_id", "offer_id", "time_received")
            .agg(
                F.sum(F.when(in_short, 1).otherwise(0)).cast("int").alias("tx_count_7d"),
                F.sum(F.when(in_long, 1).otherwise(0)).cast("int").alias("tx_count_28d"),
                F.sum(F.when(in_long, F.coalesce(F.col("amount"), F.lit(0.0))).otherwise(0.0)).alias("tx_amount_sum_28d"),
                F.max("t_event").alias("_last_tx_time"),
                F.count("*").cast("int").alias("transaction_count_before"),
                F.sum(F.coalesce(F.col("amount"), F.lit(0.0))).alias("total_spend_before"),
                F.avg(F.col("amount")).alias("avg_ticket_before"),
                F.max(F.col("amount")).alias("max_ticket_before"),
                F.min(F.col("amount")).alias("min_ticket_before"),
            )
    )

    offer_hist = (
        hist.where(F.col("event").isin(["offer received", "offer viewed", "offer completed"]))
            .withColumn("is_received", F.when(F.col("event") == "offer received", 1).otherwise(0))
            .withColumn("is_viewed", F.when(F.col("event") == "offer viewed", 1).otherwise(0))
            .withColumn("is_completed", F.when(F.col("event") == "offer completed", 1).otherwise(0))
            .groupBy("customer_id", "offer_id", "time_received")
            .agg(
                F.sum(F.when(in_long, F.col("is_received")).otherwise(0)).cast("int").alias("_offers_received_28d"),
                F.sum(F.when(in_long, F.col("is_viewed")).otherwise(0)).cast("int").alias("offers_viewed_count_28d"),
                F.sum(F.when(in_long, F.col("is_completed")).otherwise(0)).cast("int").alias("offers_completed_count_28d"),
                F.max(F.when(F.col("event") == "offer viewed", F.col("t_event"))).alias("_last_view_time"),
                F.max(F.when(F.col("event") == "offer completed", F.col("t_event"))).alias("_last_completed_time"),
                F.sum(F.col("is_received")).cast("int").alias("offers_received_count_before"),
                F.sum(F.col("is_viewed")).cast("int").alias("offers_viewed_count_before"),
                F.sum(F.col("is_completed")).cast("int").alias("offers_completed_count_before"),
            )
    )

    out = (
        t.join(tx_agg, on=["customer_id", "offer_id", "time_received"], how="left")
         .join(offer_hist, on=["customer_id", "offer_id", "time_received"], how="left")
    )

    out = (
        out.withColumn(
            "days_since_last_tx",
            F.when(F.col("_last_tx_time").isNotNull(), F.col("time_received") - F.col("_last_tx_time"))
             .otherwise(F.lit(None).cast("double"))
        )
        .withColumn(
            "days_since_last_offer_view",
            F.when(F.col("_last_view_time").isNotNull(), F.col("time_received") - F.col("_last_view_time"))
             .otherwise(F.lit(None).cast("double"))
        )
        .withColumn(
            "days_since_last_offer_completion",
            F.when(F.col("_last_completed_time").isNotNull(), F.col("time_received") - F.col("_last_completed_time"))
             .otherwise(F.lit(None).cast("double"))
        )
    )

    out = (
        out.withColumn(
            "offer_completion_rate_28d",
            F.when(F.col("_offers_received_28d") > 0, F.col("offers_completed_count_28d") / F.col("_offers_received_28d"))
             .otherwise(F.lit(0.0))
        )
        .withColumn(
            "offer_view_rate_28d",
            F.when(F.col("_offers_received_28d") > 0, F.col("offers_viewed_count_28d") / F.col("_offers_received_28d"))
             .otherwise(F.lit(0.0))
        )
        .withColumn(
            "avg_ticket_28d",
            F.when(F.col("tx_count_28d") > 0, F.col("tx_amount_sum_28d") / F.col("tx_count_28d"))
            .otherwise(F.lit(0.0))
        )
    )

    out = (
        out
        .withColumn(
            "customer_conversion_rate_before",
            F.when(
                F.col("offers_viewed_count_before") > 0,
                (F.col("offers_completed_count_before") / F.col("offers_viewed_count_before")) * 100,
            ).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "view_rate_before",
            F.when(
                F.col("offers_received_count_before") > 0,
                F.col("offers_viewed_count_before") / F.col("offers_received_count_before"),
            ).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "completion_rate_before",
            F.when(
                F.col("offers_received_count_before") > 0,
                F.col("offers_completed_count_before") / F.col("offers_received_count_before"),
            ).otherwise(F.lit(0.0)),
        )
    )

    out = out.drop("_last_tx_time", "_last_view_time", "_last_completed_time", "_offers_received_28d")
    return out


def _last_offer_viewed_type(base: DataFrame, df_joined: DataFrame) -> DataFrame:
    """Type of the last offer viewed before the current offer was received."""
    viewed_events = (
        df_joined.filter(F.col("event") == "offer viewed")
            .select(
                F.col("user_id").alias("customer_id"),
                F.col("time_since_test_start").cast("double").alias("t_view_event"),
                F.col("offer_type").cast("string").alias("viewed_offer_type"),
            )
    )

    anchor = base.select("customer_id", "offer_id", "time_received")

    joined = (
        anchor.join(viewed_events, on="customer_id", how="left")
              .where(F.col("t_view_event") < F.col("time_received"))
    )

    last_type = (
        joined.withColumn(
            "rn",
            F.row_number().over(
                Window.partitionBy("customer_id", "offer_id", "time_received")
                      .orderBy(F.col("t_view_event").desc())
            )
        )
        .where(F.col("rn") == 1)
        .select(
            "customer_id", "offer_id", "time_received",
            F.col("viewed_offer_type").alias("last_viewed_offer_type")
        )
    )

    return (
        anchor.join(last_type, on=["customer_id", "offer_id", "time_received"], how="left")
              .withColumn("last_viewed_offer_type", F.coalesce(F.col("last_viewed_offer_type"), F.lit("Unknown")))
    )


def _customer_tenure_feature(base: DataFrame, df_joined: DataFrame) -> DataFrame:
    """
    Customer tenure at the start of the experiment.
    Reference date = max(registered_on_date) + 1 day, derived from the data itself.
    """
    if "registered_on_date" in df_joined.columns:
        date_col = F.col("registered_on_date")
    else:
        date_col = F.to_date(F.col("registered_on").cast("string"), "yyyyMMdd")

    max_reg = (
        df_joined
        .select(F.max(date_col).alias("max_reg"))
        .first()["max_reg"]
    )
    ref_date = F.date_add(F.lit(max_reg), 1)

    reg = (
        df_joined
        .select(
            F.col("user_id").alias("customer_id"),
            date_col.alias("_reg_date"),
        )
        .dropDuplicates(["customer_id"])
        .withColumn("customer_tenure_days", F.datediff(ref_date, F.col("_reg_date")))
        .select("customer_id", "customer_tenure_days")
    )

    return (
        base.select("customer_id", "offer_id", "time_received")
            .join(reg, on="customer_id", how="left")
    )


def _final_null_handling(df: DataFrame) -> DataFrame:
    fill_zero = [
        "tx_count_7d",
        "tx_count_28d",
        "tx_amount_sum_28d",
        "offers_viewed_count_28d",
        "offers_completed_count_28d",
        "offer_completion_rate_28d",
        "offer_view_rate_28d",
        "offer_channel_count",
        "offer_reward_ratio",
        "transaction_count_before",
        "total_spend_before",
        "offers_received_count_before",
        "offers_viewed_count_before",
        "offers_completed_count_before",
        "customer_conversion_rate_before",
        "view_rate_before",
        "completion_rate_before",
    ]
    for c in fill_zero:
        if c in df.columns:
            df = df.withColumn(c, F.coalesce(F.col(c), F.lit(0.0)))

    for c in ["customer_gender", "offer_duration_bucket", "last_viewed_offer_type", "offer_type"]:
        if c in df.columns:
            df = df.withColumn(c, F.coalesce(F.col(c), F.lit("Unknown")))

    return df


def _infer_column_types(df: DataFrame) -> Tuple[List[str], List[str]]:
    exclude = {"customer_id", "offer_id", "time_received", "time_expires", "target"}
    numerical_types = {"int", "bigint", "float", "double", "decimal", "short", "byte", "long"}
    categorical_types = {"string", "boolean"}

    numerical_cols: List[str] = []
    categorical_cols: List[str] = []

    for col_name, dtype in df.dtypes:
        if col_name in exclude:
            continue
        if dtype in numerical_types:
            numerical_cols.append(col_name)
        elif dtype in categorical_types:
            categorical_cols.append(col_name)

    return numerical_cols, categorical_cols
