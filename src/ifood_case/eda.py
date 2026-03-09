# src/ifood_case/eda.py

from __future__ import annotations

from typing import Optional, List

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


# -----------------------------
# Styling / Orders
# -----------------------------
IFOOD_RED   = "#EA1D2C"
IFOOD_BLACK = "#3F3E3E"

COLOR_BOGO           = "#1F3A8A"
COLOR_DISCOUNT       = "#2E7D32"
COLOR_INFORMATIONAL  = "#7f7f7f"

PALETTE_OFFER = {
    "bogo":          COLOR_BOGO,
    "discount":      COLOR_DISCOUNT,
    "informational": COLOR_INFORMATIONAL,
}

OFFER_ORDER  = ["bogo", "discount", "informational"]
GENDER_ORDER = ["F", "M", "O", "Unknown"]
AGE_ORDER    = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]


# -----------------------------
# Data Quality
# -----------------------------
def plot_null_summary(df: DataFrame, cols: list, title: str = "Null Rate by Column") -> None:
    """Horizontal bar chart of null % per column. Green <5 %, yellow 5-20 %, red >20 %."""
    total = df.count()
    rows = []
    for c in cols:
        n_null = df.filter(F.col(c).isNull()).count()
        rows.append({"column": c, "null_pct": round(100 * n_null / total, 1)})
    pdf = pd.DataFrame(rows).sort_values("null_pct", ascending=True)

    colors = [
        IFOOD_RED if v > 20 else ("#F9A825" if v > 5 else "#2E7D32")
        for v in pdf["null_pct"]
    ]

    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.45 * len(cols))))
    bars = ax.barh(pdf["column"], pdf["null_pct"], color=colors, edgecolor="white")

    for bar, val in zip(bars, pdf["null_pct"]):
        ax.text(
            bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{val}%", va="center", fontsize=9, color=IFOOD_BLACK,
        )

    ax.axvline(5, color="#F9A825", linewidth=1, linestyle="--", alpha=0.7, label="5 % threshold")
    ax.set_xlabel("Null rate (%)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", color=IFOOD_BLACK)
    ax.set_xlim(0, max(pdf["null_pct"].max() * 1.25, 8))
    ax.legend(fontsize=8)
    sns.despine()
    plt.tight_layout()
    plt.show()


# -----------------------------
# Helpers
# -----------------------------
def _annotate_percent(ax, decimals: int = 1):
    for p in ax.patches:
        h = p.get_height()
        if h is None or (isinstance(h, float) and pd.isna(h)):
            continue
        ax.annotate(
            f"{h * 100:.{decimals}f}%",
            (p.get_x() + p.get_width() / 2, h),
            ha="center", va="bottom",
            fontsize=10, fontweight="bold",
            xytext=(0, 3), textcoords="offset points",
        )


def _annotate_int(ax):
    for p in ax.patches:
        h = p.get_height()
        if h is None or (isinstance(h, float) and pd.isna(h)):
            continue
        ax.annotate(
            f"{int(h):,}".replace(",", "."),
            (p.get_x() + p.get_width() / 2, h),
            ha="center", va="bottom",
            fontsize=9,
            xytext=(0, 3), textcoords="offset points",
        )


# =========================================================
# Univariate
# =========================================================

def plot_event_funnel(df_joined: DataFrame):
    """
    Bar chart of event volumes with offer funnel proportions.
    Shows organic transactions vs offer engagement stages.
    """
    funnel_order = ["offer received", "offer viewed", "offer completed"]

    counts = {
        row["event"]: row["count"]
        for row in df_joined
            .groupBy("event").count()
            .filter(F.col("event").isin(funnel_order))
            .collect()
    }
    tx_count = df_joined.filter(F.col("event") == "transaction").count()

    received        = counts.get("offer received",  0)
    viewed          = counts.get("offer viewed",    0)
    completed       = counts.get("offer completed", 0)

    view_rate       = viewed    / received if received else 0
    completion_rate = completed / received if received else 0

    plot_df = pd.DataFrame({
        "event": ["transaction", "offer received", "offer viewed", "offer completed"],
        "count": [tx_count, received, viewed, completed],
    })

    labels = {
        "transaction":     f"{tx_count:,}\n(organic)",
        "offer received":  f"{received:,}\n(100%)",
        "offer viewed":    f"{viewed:,}\n({view_rate:.1%})",
        "offer completed": f"{completed:,}\n({completion_rate:.1%})",
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(plot_df["event"], plot_df["count"], color=[IFOOD_BLACK, IFOOD_RED, "#FF7043", "#8B0000"])

    for bar, ev in zip(bars, plot_df["event"]):
        h = bar.get_height()
        ax.annotate(labels[ev], (bar.get_x() + bar.get_width() / 2, h),
                    ha="center", va="bottom", fontsize=10, xytext=(0, 5),
                    textcoords="offset points")

    ax.set_title("Offer Engagement Funnel", fontsize=13)
    ax.set_ylabel("Count")
    ax.set_xlabel("")
    plt.xticks(rotation=20, ha="right")
    plt.figtext(0.99, 0.01,
                f"Completed / Received: {completion_rate:.1%} — model opportunity",
                ha="right", fontsize=9, color=IFOOD_BLACK)
    plt.tight_layout()
    plt.show()


def plot_age_distribution(profile_raw: DataFrame):
    """
    Histogram + box plot of customer age (raw data, 1 row per customer).
    Uses profile_raw to reflect the true distribution before any processing.
    """
    pdf = (
        profile_raw
        .select(F.col("id").alias("user_id"), "age")
        .dropDuplicates(["user_id"])
        .filter(F.col("age").isNotNull())
        .toPandas()
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    sns.histplot(data=pdf, x="age", bins=30, color=IFOOD_RED, edgecolor="white", ax=axes[0])
    axes[0].set_title("Age Distribution")
    axes[0].set_xlabel("Age")
    axes[0].set_ylabel("Customers")

    sns.boxplot(x=pdf["age"], color=IFOOD_RED, ax=axes[1])
    axes[1].set_title("Age — Box Plot")
    axes[1].set_xlabel("Age")

    plt.suptitle("Univariate — Age", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_credit_limit_distribution(profile_raw: DataFrame):
    """
    Histogram + box plot of credit limit (raw data, 1 row per customer).
    Uses profile_raw to reflect the true distribution before any processing.
    """
    pdf = (
        profile_raw
        .select(F.col("id").alias("user_id"), "credit_card_limit")
        .dropDuplicates(["user_id"])
        .filter(F.col("credit_card_limit").isNotNull())
        .toPandas()
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    sns.histplot(data=pdf, x="credit_card_limit", bins=30,
                 color=IFOOD_RED, edgecolor="white", ax=axes[0])
    axes[0].set_title("Credit Limit Distribution")
    axes[0].set_xlabel("Limit (BRL)")
    axes[0].set_ylabel("Customers")

    sns.boxplot(x=pdf["credit_card_limit"], color=IFOOD_RED, ax=axes[1])
    axes[1].set_title("Credit Limit — Box Plot")
    axes[1].set_xlabel("Limit (BRL)")

    plt.suptitle("Univariate — Credit Limit", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_gender_distribution(profile_raw: DataFrame):
    """
    Bar chart of gender distribution (% per unique customer).
    Uses profile_raw to reflect the true distribution before any processing.
    """
    total = profile_raw.select("id").dropDuplicates(["id"]).count()

    pdf = (
        profile_raw
        .select(F.col("id").alias("user_id"), "gender")
        .dropDuplicates(["user_id"])
        .withColumn("gender", F.coalesce(F.col("gender"), F.lit("Unknown")))
        .groupBy("gender").count()
        .withColumn("pct", F.col("count") / F.lit(total))
        .toPandas()
    )
    pdf["gender"] = pd.Categorical(pdf["gender"], categories=GENDER_ORDER, ordered=True)
    pdf = pdf.sort_values("gender")

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(pdf["gender"], pdf["pct"], color=IFOOD_RED)
    for bar, v in zip(bars, pdf["pct"]):
        ax.text(bar.get_x() + bar.get_width() / 2, v,
                f"{v:.1%}", ha="center", va="bottom", fontsize=10)
    ax.set_title("Gender Distribution")
    ax.set_xlabel("")
    ax.set_ylabel("% of Customers")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    plt.tight_layout()
    plt.show()


def plot_channel_distribution(offers: DataFrame):
    """
    Bar chart of individual channels (explode from channels array).
    """
    pdf = (
        offers
        .withColumn("channel", F.explode(F.col("channels")))
        .groupBy("channel").count()
        .orderBy(F.desc("count"))
        .toPandas()
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=pdf, x="channel", y="count", color=IFOOD_RED, ax=ax)
    _annotate_int(ax)
    ax.set_title("Individual Channel Distribution")
    ax.set_xlabel("")
    ax.set_ylabel("# of Offers")
    plt.tight_layout()
    plt.show()


def plot_offer_type_distribution(opps_enriched: DataFrame):
    """
    Bar chart of received share by offer_type.
    """
    total = opps_enriched.count()
    pdf = (
        opps_enriched
        .groupBy("offer_type").count()
        .withColumn("share", F.col("count") / F.lit(total))
        .toPandas()
    )
    pdf["offer_type"] = pd.Categorical(pdf["offer_type"], categories=OFFER_ORDER, ordered=True)
    pdf = pdf.sort_values("offer_type")

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = [ax.bar(row["offer_type"], row["share"],
                   color=PALETTE_OFFER.get(row["offer_type"], IFOOD_RED))
            for _, row in pdf.iterrows()]
    for bar_list, v in zip(bars, pdf["share"]):
        b = bar_list[0]
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1%}",
                ha="center", va="bottom", fontsize=10)
    ax.set_title("Received Share by Offer Type")
    ax.set_xlabel("")
    ax.set_ylabel("Share")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    plt.tight_layout()
    plt.show()


def plot_target_distribution(opps_enriched: DataFrame):
    """
    Bar chart of overall conversion rate (class balance).
    Each row in opps_enriched is one offer opportunity (user × offer × received_time).
    The proportion shows the opportunity-level conversion rate, not unique customer rate.
    """
    pdf = opps_enriched.groupBy("completed").count().toPandas()
    total = pdf["count"].sum()
    pdf["rate"] = pdf["count"] / total

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(pdf["completed"].astype(str), pdf["rate"], color=[IFOOD_BLACK, IFOOD_RED])
    for bar, v in zip(bars, pdf["rate"]):
        ax.text(bar.get_x() + bar.get_width() / 2, v,
                f"{v:.1%}", ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_title("Target Distribution — Overall Conversion Rate")
    ax.set_xlabel("Completed (0 = Did not convert, 1 = Converted)")
    ax.set_ylabel("Proportion")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    plt.tight_layout()
    plt.show()


# =========================================================
# Bivariate
# =========================================================

def plot_age_vs_credit_limit(df_joined: DataFrame, sample_fraction: float = 0.08):
    """
    Scatter (sample) + box plot of credit limit by age group.
    Shows relationship between age and purchasing power.
    """
    pdf = (
        df_joined
        .select("user_id", "age", "credit_card_limit", "age_group")
        .dropDuplicates(["user_id"])
        .filter(F.col("age").isNotNull() & F.col("credit_card_limit").isNotNull())
        .sample(False, sample_fraction, seed=42)
        .toPandas()
    )

    pdf_box = (
        df_joined
        .select("user_id", "age_group", "credit_card_limit")
        .dropDuplicates(["user_id"])
        .filter(F.col("age_group").isNotNull() & F.col("credit_card_limit").isNotNull())
        .filter(F.col("age_group") != "Unknown")
        .toPandas()
    )
    pdf_box["age_group"] = pd.Categorical(pdf_box["age_group"], categories=AGE_ORDER, ordered=True)
    pdf_box = pdf_box.sort_values("age_group")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].scatter(pdf["age"], pdf["credit_card_limit"], alpha=0.2, color=IFOOD_RED, s=10)
    axes[0].set_title("Age vs Credit Limit (income proxy)")
    axes[0].set_xlabel("Age")
    axes[0].set_ylabel("Credit Limit (BRL)")

    sns.boxplot(data=pdf_box, x="age_group", y="credit_card_limit",
                color=IFOOD_RED, ax=axes[1])
    axes[1].set_title("Credit Limit by Age Group")
    axes[1].set_xlabel("Age Group")
    axes[1].set_ylabel("Credit Limit (BRL)")
    axes[1].tick_params(axis="x", rotation=25)

    plt.suptitle("Bivariate — Age × Credit Limit", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_q2_group_offer_duo(
    opps_enriched: DataFrame,
    group_col: str,
    group_order: Optional[List[str]] = None,
):
    """
    Q2: Do different groups react differently to offers?

    Two side-by-side panels:
      - Completed / Received by group_col and offer_type
      - Viewed / Received by group_col and offer_type
    """
    df = opps_enriched
    if group_col == "gender":
        df = df.withColumn("gender", F.coalesce(F.col("gender"), F.lit("Unknown")))

    agg = (
        df
        .groupBy(group_col, "offer_type")
        .agg(
            F.count("*").alias("n_received"),
            F.sum("completed").alias("n_completed"),
            F.sum("viewed").alias("n_viewed"),
        )
        .withColumn("completed_rate", F.col("n_completed") / F.col("n_received"))
        .withColumn("viewed_rate",    F.col("n_viewed")    / F.col("n_received"))
    )

    pdf = agg.toPandas()
    pdf["offer_type"] = pd.Categorical(pdf["offer_type"], categories=OFFER_ORDER, ordered=True)
    if group_order:
        pdf[group_col] = pd.Categorical(pdf[group_col], categories=group_order, ordered=True)
    pdf = pdf.sort_values([group_col, "offer_type"])

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

    ax0 = sns.barplot(data=pdf, x=group_col, y="completed_rate",
                      hue="offer_type", hue_order=OFFER_ORDER,
                      palette=PALETTE_OFFER, ax=axes[0])
    axes[0].set_title(f"Completed / Received — by {group_col}")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Conversion Rate")
    axes[0].yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    axes[0].tick_params(axis="x", rotation=25)
    _annotate_percent(ax0)
    axes[0].legend(title="offer_type")

    ax1 = sns.barplot(data=pdf, x=group_col, y="viewed_rate",
                      hue="offer_type", hue_order=OFFER_ORDER,
                      palette=PALETTE_OFFER, ax=axes[1])
    axes[1].set_title(f"Viewed / Received — by {group_col}")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")
    axes[1].yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    axes[1].tick_params(axis="x", rotation=25)
    _annotate_percent(ax1)
    if axes[1].get_legend():
        axes[1].get_legend().remove()

    plt.tight_layout()
    plt.show()
    return agg


def plot_duration_conversion_bar(opps_enriched: DataFrame):
    """
    Completed/Received and Viewed/Received by offer duration (in days).
    Connects to SHAP: offer_duration_days is a relevant driver.
    """
    agg = (
        opps_enriched
        .groupBy("duration")
        .agg(
            F.count("*").alias("n_received"),
            F.sum("viewed").alias("n_viewed"),
            F.sum("completed").alias("n_completed"),
        )
        .withColumn("viewed_rate",    F.col("n_viewed")    / F.col("n_received"))
        .withColumn("completed_rate", F.col("n_completed") / F.col("n_received"))
        .orderBy("duration")
    )

    pdf = agg.toPandas()
    x = pdf["duration"].astype(str)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, pdf["completed_rate"], label="Completed/Received", color=IFOOD_RED, alpha=0.85)
    ax.bar(x, pdf["viewed_rate"],    label="Viewed/Received",    color=IFOOD_BLACK, alpha=0.35)

    for i, v in enumerate(pdf["completed_rate"]):
        ax.text(i, v + 0.005, f"{v:.1%}", ha="center", va="bottom", fontsize=10)

    ax.set_title("Conversion by Offer Duration")
    ax.set_xlabel("Duration (days)")
    ax.set_ylabel("Rate")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend()
    plt.tight_layout()
    plt.show()
    return agg


# =========================================================
# Multivariate
# =========================================================

_METRICS       = ["completed_received", "viewed_received", "completed_viewed"]
_METRIC_LABELS = ["Completed/Received", "Viewed/Received", "Completed/Viewed"]


def plot_q1_funnel_by_offer_type(opps_enriched: DataFrame):
    """
    Q1: Offer funnel by offer type — 2 panels (bogo/discount | informational).
    """
    funnel = (
        opps_enriched
        .groupBy("offer_type")
        .agg(
            F.count("*").alias("n_received"),
            F.sum("viewed").alias("n_viewed"),
            F.sum("completed").alias("n_completed"),
        )
        .withColumn("completed_received", F.col("n_completed") / F.col("n_received"))
        .withColumn("viewed_received",    F.col("n_viewed")    / F.col("n_received"))
        .withColumn("completed_viewed",   F.when(F.col("n_viewed") > 0,
                                                 F.col("n_completed") / F.col("n_viewed")))
    )

    pdf = funnel.toPandas()
    bd   = pdf[pdf["offer_type"].isin(["bogo", "discount"])].copy()
    info = pdf[pdf["offer_type"] == "informational"].copy()

    def _melt(df):
        m = df.melt(id_vars="offer_type", value_vars=_METRICS,
                    var_name="metric", value_name="rate")
        m["metric"] = pd.Categorical(m["metric"], categories=_METRICS, ordered=True)
        return m.sort_values(["metric", "offer_type"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    ax0 = sns.barplot(data=_melt(bd), x="metric", y="rate", hue="offer_type",
                      palette=PALETTE_OFFER, ax=axes[0])
    axes[0].set_title("BOGO vs Discount")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Rate")
    axes[0].yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    axes[0].set_xticklabels(_METRIC_LABELS, rotation=20, ha="right")
    _annotate_percent(ax0)
    axes[0].legend(title="offer_type")

    ax1 = sns.barplot(data=_melt(info), x="metric", y="rate", hue="offer_type",
                      palette=PALETTE_OFFER, ax=axes[1])
    axes[1].set_title("Informational")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")
    axes[1].yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    axes[1].set_xticklabels(_METRIC_LABELS, rotation=20, ha="right")
    _annotate_percent(ax1)
    if axes[1].get_legend():
        axes[1].get_legend().remove()

    plt.tight_layout()
    plt.show()
    return funnel


def plot_offer_effectiveness_by_profile(opps_enriched: DataFrame):
    """
    Multivariate: completed_rate by offer_type × gender × age_group.
    Catplot with one panel per gender.
    """
    df = opps_enriched.withColumn("gender", F.coalesce(F.col("gender"), F.lit("Unknown")))

    agg = (
        df
        .groupBy("offer_type", "gender", "age_group")
        .agg(
            F.count("*").alias("n_received"),
            F.sum("completed").alias("n_completed"),
        )
        .withColumn("completed_rate", F.col("n_completed") / F.col("n_received"))
        .filter(F.col("n_received") >= 10)
    )

    pdf = agg.toPandas()
    pdf["offer_type"] = pd.Categorical(pdf["offer_type"], categories=OFFER_ORDER, ordered=True)
    pdf["age_group"]  = pd.Categorical(pdf["age_group"],  categories=AGE_ORDER,   ordered=True)
    pdf["gender"]     = pd.Categorical(pdf["gender"],     categories=GENDER_ORDER, ordered=True)
    pdf = pdf.sort_values(["gender", "age_group", "offer_type"])

    g = sns.catplot(
        data=pdf,
        x="age_group", y="completed_rate",
        hue="offer_type", col="gender",
        hue_order=OFFER_ORDER, col_order=GENDER_ORDER,
        palette=PALETTE_OFFER,
        kind="bar", height=5, aspect=1.1,
        sharey=True,
    )
    g.fig.suptitle("Multivariate — Conversion by Offer Type × Gender × Age Group",
                   y=1.02, fontsize=13)
    g.set_axis_labels("Age Group", "Completed / Received")
    g.set_titles("Gender: {col_name}")
    for ax in g.axes.flat:
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.show()
    return agg


def plot_conversion_by_gender_age(opps_enriched: DataFrame):
    """
    Multivariate heatmap: conversion rate by gender × age_group.
    Replaces separate group_offer_duo calls with a single compact view.
    """
    df = opps_enriched.withColumn("gender", F.coalesce(F.col("gender"), F.lit("Unknown")))

    agg = (
        df
        .groupBy("gender", "age_group")
        .agg(
            F.count("*").alias("n_received"),
            F.sum("completed").alias("n_completed"),
        )
        .withColumn("conversion_rate", F.col("n_completed") / F.col("n_received"))
        .filter(F.col("n_received") >= 10)
    )

    pdf = agg.toPandas()
    pivot = pdf.pivot(index="gender", columns="age_group", values="conversion_rate")
    pivot = pivot.reindex(index=GENDER_ORDER, columns=AGE_ORDER)

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(pivot, annot=True, fmt=".1%", cmap="Reds",
                linewidths=0.5, ax=ax, cbar_kws={"label": "Conversion Rate"})
    ax.set_title("Conversion Rate — Gender × Age Group", fontsize=13)
    ax.set_xlabel("Age Group")
    ax.set_ylabel("Gender")
    plt.tight_layout()
    plt.show()
    return pdf


def plot_channel_conversion(opps_enriched: DataFrame, by_offer_type: bool = True):
    """
    Completed/Received by channel, segmented by offer_type.
    """
    base = (
        opps_enriched
        .filter(F.col("channels").isNotNull())
        .withColumn("channel", F.explode(F.col("channels")))
    )

    if by_offer_type:
        agg = (
            base.groupBy("channel", "offer_type")
            .agg(
                F.count("*").alias("n_received"),
                F.sum("completed").alias("n_completed"),
            )
            .withColumn("completed_rate", F.col("n_completed") / F.col("n_received"))
        )
        pdf = agg.toPandas()
        pdf["offer_type"] = pd.Categorical(pdf["offer_type"], categories=OFFER_ORDER, ordered=True)

        fig, ax = plt.subplots(figsize=(12, 5))
        sns.barplot(data=pdf, x="channel", y="completed_rate",
                    hue="offer_type", hue_order=OFFER_ORDER,
                    palette=PALETTE_OFFER, ax=ax)
        ax.set_title("Completed/Received by Channel and Offer Type")
        ax.set_xlabel("")
        ax.set_ylabel("Completed/Received")
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.tick_params(axis="x", rotation=25)
        _annotate_percent(ax)
        plt.tight_layout()
        plt.show()
        return agg

    agg = (
        base.groupBy("channel")
        .agg(
            F.count("*").alias("n_received"),
            F.sum("completed").alias("n_completed"),
        )
        .withColumn("completed_rate", F.col("n_completed") / F.col("n_received"))
        .orderBy(F.desc("completed_rate"))
    )
    pdf = agg.toPandas()

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=pdf, x="channel", y="completed_rate", color=IFOOD_RED, ax=ax)
    _annotate_percent(ax)
    ax.set_title("Completed/Received by Channel")
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    plt.tight_layout()
    plt.show()
    return agg


# =========================================================
# Offer Engagement Path
# =========================================================

def plot_q3_view_then_use(opps_enriched: DataFrame):
    """
    Q3: Do customers typically view an offer before completing, or complete without viewing?
    """
    agg = (
        opps_enriched
        .filter(F.col("completed") == 1)
        .withColumn("without_view", F.when(F.col("viewed") == 0, 1).otherwise(0))
        .agg(
            F.count("*").alias("total_completed"),
            F.sum("without_view").alias("n_without_view"),
        )
        .withColumn("rate_without_view", F.col("n_without_view") / F.col("total_completed"))
    )

    row = agg.toPandas().iloc[0]
    without = float(row["rate_without_view"])
    with_v  = 1.0 - without

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(["Viewed → Completed", "Completed without View"],
                  [with_v, without], color=[IFOOD_RED, IFOOD_BLACK])
    for bar, v in zip(bars, [with_v, without]):
        ax.text(bar.get_x() + bar.get_width() / 2, v,
                f"{v:.1%}", ha="center", va="bottom", fontweight="bold")
    ax.set_title("Q3: Does completion follow a view or happen organically?")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    plt.tight_layout()
    plt.show()
    return agg


def plot_q4_group_use_without_view(
    opps_enriched: DataFrame,
    group_col: str,
    group_order: Optional[List[str]] = None,
):
    """
    Q4: Which groups complete offers most often without viewing?
    Metric: P(completed=1 AND viewed=0) / P(received)
    """
    df = opps_enriched
    if group_col == "gender":
        df = df.withColumn("gender", F.coalesce(F.col("gender"), F.lit("Unknown")))

    agg = (
        df
        .withColumn("used_without_view",
                    F.when((F.col("completed") == 1) & (F.col("viewed") == 0), 1).otherwise(0))
        .groupBy(group_col)
        .agg(
            F.count("*").alias("n_received"),
            F.sum("used_without_view").alias("n_without_view"),
        )
        .withColumn("rate", F.col("n_without_view") / F.col("n_received"))
        .orderBy(F.desc("rate"))
    )

    pdf = agg.toPandas()
    if group_order:
        pdf[group_col] = pd.Categorical(pdf[group_col], categories=group_order, ordered=True)
        pdf = pdf.sort_values(group_col)

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(data=pdf, x=group_col, y="rate", color=IFOOD_RED, ax=ax)
    _annotate_percent(ax)
    ax.set_title(f"Q4: Completed without View — by {group_col}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    plt.show()
    return agg


def plot_q5_offer_type_used_without_view(opps_enriched: DataFrame):
    """
    Q5: Which offer type is most completed without viewing?
    """
    agg = (
        opps_enriched
        .withColumn("used_without_view",
                    F.when((F.col("completed") == 1) & (F.col("viewed") == 0), 1).otherwise(0))
        .groupBy("offer_type")
        .agg(
            F.count("*").alias("n_received"),
            F.sum("used_without_view").alias("n_without_view"),
        )
        .withColumn("rate", F.col("n_without_view") / F.col("n_received"))
        .orderBy(F.desc("rate"))
    )

    pdf = agg.toPandas()
    pdf["offer_type"] = pd.Categorical(pdf["offer_type"], categories=OFFER_ORDER, ordered=True)
    pdf = pdf.sort_values("offer_type")

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=pdf, x="offer_type", y="rate",
                order=OFFER_ORDER, palette=PALETTE_OFFER, ax=ax)
    _annotate_percent(ax)
    ax.set_title("Q5: Which offer type is most completed without viewing?")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    plt.tight_layout()
    plt.show()
    return agg


# =========================================================
# Spending Analysis
# =========================================================

def plot_transaction_value_step_log(df_joined: DataFrame):
    """
    Transaction value distribution (log scale) with step + KDE.
    Compares users who completed at least one offer vs those who did not.
    Connects to SHAP: avg_ticket_before is the top model driver.
    """
    users_used = (
        df_joined
        .filter(F.col("event") == "offer completed")
        .select("user_id").distinct()
        .withColumn("uses_offers", F.lit(1))
    )

    tx = (
        df_joined
        .filter(F.col("event") == "transaction")
        .select("user_id", F.col("amount").cast("double").alias("amount"))
        .filter(F.col("amount").isNotNull() & (F.col("amount") > 0))
        .join(users_used, on="user_id", how="left")
        .withColumn("uses_offers", F.coalesce(F.col("uses_offers"), F.lit(0)))
        .withColumn("user_type",
                    F.when(F.col("uses_offers") == 1, "Uses Offers")
                     .otherwise("Does Not Use Offers"))
    )

    pdf = tx.sample(False, 0.3, seed=42).toPandas()

    summary = (
        tx.groupBy("uses_offers")
          .agg(
              F.expr("percentile_approx(amount, 0.5)").alias("median"),
              F.avg("amount").alias("mean"),
              F.count("*").alias("n"),
          )
          .orderBy("uses_offers")
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(
        data=pdf, x="amount", hue="user_type",
        palette={"Uses Offers": IFOOD_RED, "Does Not Use Offers": IFOOD_BLACK},
        log_scale=True, element="step", kde=True, ax=ax,
    )
    ax.set_title("Transaction Value Distribution (Log Scale)\nOffer Users vs Non-Users")
    ax.set_xlabel("Transaction Value (BRL) — Log Scale")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.show()

    return summary


def plot_box_transaction_by_target(df_joined: DataFrame):
    """
    Box plot (log scale) of transaction value by offer usage profile.
    Shows that converters have a higher ticket distribution.
    """
    users_used = (
        df_joined
        .filter(F.col("event") == "offer completed")
        .select("user_id").distinct()
        .withColumn("uses_offer", F.lit(1))
    )

    tx = (
        df_joined
        .filter(F.col("event") == "transaction")
        .select("user_id", F.col("amount").cast("double").alias("amount"))
        .filter(F.col("amount") > 0)
        .join(users_used, on="user_id", how="left")
        .withColumn("uses_offer", F.coalesce(F.col("uses_offer"), F.lit(0)))
    )

    pdf = tx.sample(False, 0.3, seed=42).toPandas()
    pdf["Profile"] = pdf["uses_offer"].map({1: "Uses Offer", 0: "Does Not Use Offer"})

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=pdf, x="Profile", y="amount", palette=[IFOOD_RED, IFOOD_BLACK], ax=ax)
    ax.set_yscale("log")
    ax.set_title("Transaction Value by Offer Usage Profile (log)")
    ax.set_xlabel("")
    ax.set_ylabel("Transaction Value (BRL)")
    plt.tight_layout()
    plt.show()
    return tx


# =========================================================
# Temporal Patterns
# =========================================================

def plot_conversion_over_time(opps_enriched: DataFrame):
    """
    Conversion rate over the experiment timeline (by t_received).
    Identifies peaks, drops, and possible campaign fatigue.
    """
    agg = (
        opps_enriched
        .groupBy("t_received")
        .agg(
            F.count("*").alias("n"),
            F.sum("completed").alias("n_completed"),
        )
        .withColumn("conversion_rate", F.col("n_completed") / F.col("n"))
        .orderBy("t_received")
    )

    pdf = agg.toPandas()

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(pdf["t_received"], pdf["conversion_rate"], color=IFOOD_RED, lw=2)
    ax.fill_between(pdf["t_received"], pdf["conversion_rate"], alpha=0.15, color=IFOOD_RED)
    ax.set_title("Conversion Rate Over Experiment Timeline")
    ax.set_xlabel("Time since test start (hours)")
    ax.set_ylabel("Conversion Rate")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()
    return agg


def plot_engagement_by_week(df_joined: DataFrame):
    """
    Volume of viewed and completed events per experiment week,
    segmented by gender and age_group.
    Identifies trends and campaign fatigue by profile.
    Note: filters out rows where time_since_test_start is null.
    """
    events_weekly = (
        df_joined
        .filter(F.col("event").isin("offer viewed", "offer completed"))
        .filter(F.col("time_since_test_start").isNotNull())
        .withColumn("week", (F.col("time_since_test_start") / 7).cast("int"))
        .withColumn("gender", F.coalesce(F.col("gender"), F.lit("Unknown")))
    )

    by_gender = (
        events_weekly
        .groupBy("week", "gender", "event")
        .count()
        .orderBy("week", "gender")
        .toPandas()
    )

    by_age = (
        events_weekly
        .groupBy("week", "age_group", "event")
        .count()
        .orderBy("week", "age_group")
        .toPandas()
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    sns.lineplot(data=by_gender, x="week", y="count",
                 hue="gender", style="event",
                 hue_order=GENDER_ORDER, lw=2, ax=axes[0])
    axes[0].set_title("Engagement by Week × Gender")
    axes[0].set_xlabel("Experiment week")
    axes[0].set_ylabel("# of events")
    axes[0].grid(axis="y", alpha=0.25)

    sns.lineplot(data=by_age, x="week", y="count",
                 hue="age_group", style="event",
                 hue_order=AGE_ORDER, lw=2, ax=axes[1])
    axes[1].set_title("Engagement by Week × Age Group")
    axes[1].set_xlabel("Experiment week")
    axes[1].set_ylabel("")
    axes[1].grid(axis="y", alpha=0.25)

    plt.suptitle("Temporal — Engagement Evolution", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.show()
    return by_gender, by_age


def plot_registration_cohort(df_joined: DataFrame, opps_enriched: DataFrame):
    """
    Dual-axis chart: customer registration cohort (by month) showing
    # customers (bars) and average conversion rate (line).
    Reveals whether registration vintage predicts conversion propensity.
    """
    reg = (
        df_joined
        .select("user_id", "registered_on_date")
        .dropDuplicates(["user_id"])
        .filter(F.col("registered_on_date").isNotNull())
        .withColumn("reg_month", F.date_format("registered_on_date", "yyyy-MM"))
    )

    spend = (
        df_joined
        .filter(F.col("event") == "transaction")
        .groupBy("user_id")
        .agg(F.sum(F.col("amount").cast("double")).alias("total_spend"))
    )

    conv = (
        opps_enriched
        .groupBy("user_id")
        .agg(
            F.count("*").alias("n_offers"),
            F.sum("completed").alias("n_completed"),
        )
        .withColumn("conv_rate", F.col("n_completed") / F.col("n_offers"))
    )

    cohort = (
        reg
        .join(spend, on="user_id", how="left")
        .join(conv, on="user_id", how="left")
        .groupBy("reg_month")
        .agg(
            F.count("user_id").alias("n_customers"),
            F.avg("total_spend").alias("avg_spend"),
            F.avg("conv_rate").alias("avg_conversion_rate"),
        )
        .orderBy("reg_month")
    )
    pdf = cohort.toPandas()

    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax2 = ax1.twinx()
    ax1.bar(pdf["reg_month"], pdf["n_customers"], color=IFOOD_RED, alpha=0.6, label="# Customers")
    ax2.plot(pdf["reg_month"], pdf["avg_conversion_rate"],
             color=IFOOD_BLACK, lw=2, marker="o", label="Avg Conversion Rate")
    ax1.set_xlabel("Registration Month")
    ax1.set_ylabel("# Customers", color=IFOOD_RED)
    ax2.set_ylabel("Avg Conversion Rate", color=IFOOD_BLACK)
    ax2.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    tick_step = max(1, len(pdf) // 6)
    tick_positions = list(range(0, len(pdf), tick_step))
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels([pdf["reg_month"].iloc[i] for i in tick_positions], rotation=45, ha="right")
    ax1.set_title("Customer Registration Cohort — Volume & Conversion Rate")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    plt.tight_layout()
    plt.show()
    return pdf


# =========================================================
# Key Predictive Signals (Bridge to Model)
# =========================================================

def plot_recency_vs_conversion(df_joined: DataFrame, opps_enriched: DataFrame):
    """
    Recency of last transaction BEFORE the received offer vs conversion rate.

    For each opportunity, calculates how many days elapsed since the customer's
    last transaction before receiving the offer.
    Recent customers tend to convert more — EDA validation of the recency driver.
    """
    tx_times = (
        df_joined
        .filter(F.col("event") == "transaction")
        .select("user_id", F.col("time_since_test_start").alias("tx_time"))
    )

    opps = opps_enriched.select("user_id", "offer_id",
                                F.col("t_received").alias("received_time"),
                                "completed")

    # Compute last transaction before each offer (preserving opps with no prior tx)
    tx_before = (
        opps.alias("o")
        .join(tx_times.alias("t"), on="user_id", how="left")
        .filter(F.col("t.tx_time") < F.col("o.received_time"))
        .groupBy("o.user_id", "o.offer_id", "o.received_time", "o.completed")
        .agg(F.max("t.tx_time").alias("last_tx_time"))
    )

    last_tx = (
        opps
        .join(tx_before.select("user_id", "offer_id", "received_time", "last_tx_time"),
              on=["user_id", "offer_id", "received_time"], how="left")
        .withColumn("recency_days",
                    (F.col("received_time") - F.col("last_tx_time")))
    )

    bin_order = ["0-1 day", "1-3 days", "3-7 days", "7-14 days",
                 "14-30 days", "30+ days", "No history"]

    binned = last_tx.withColumn(
        "recency_bin",
        F.when(F.col("last_tx_time").isNull(),       "No history")
         .when(F.col("recency_days") <= 1,           "0-1 day")
         .when(F.col("recency_days") <= 3,           "1-3 days")
         .when(F.col("recency_days") <= 7,           "3-7 days")
         .when(F.col("recency_days") <= 14,          "7-14 days")
         .when(F.col("recency_days") <= 30,          "14-30 days")
         .otherwise(                                  "30+ days"),
    )

    agg = (
        binned
        .groupBy("recency_bin")
        .agg(
            F.count("*").alias("n_offers"),
            F.sum("completed").alias("n_completed"),
        )
        .withColumn("conversion_rate", F.col("n_completed") / F.col("n_offers"))
    )

    pdf = agg.toPandas()
    # Ensure all bins appear even if count = 0
    full = pd.DataFrame({"recency_bin": bin_order})
    pdf = full.merge(pdf, on="recency_bin", how="left").fillna(
        {"n_offers": 0, "n_completed": 0, "conversion_rate": 0}
    )
    pdf["recency_bin"] = pd.Categorical(pdf["recency_bin"], categories=bin_order, ordered=True)
    pdf = pdf.sort_values("recency_bin")

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(pdf["recency_bin"], pdf["conversion_rate"], color=IFOOD_RED)
    for bar, v in zip(bars, pdf["conversion_rate"]):
        if pd.notna(v) and v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, v,
                    f"{v:.1%}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Recency of Last Transaction Before Offer × Conversion Rate\n"
                 "(Validates recency driver — top SHAP feature)")
    ax.set_xlabel("Recency before offer")
    ax.set_ylabel("Conversion Rate")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    plt.show()
    return agg


def plot_conversion_by_registration_quartile(df_joined: DataFrame, opps_enriched: DataFrame):
    """
    Conversion rate by customer registration quartile.
    Q1 = earliest registered customers, Q4 = most recently registered.
    Validates whether customer vintage predicts conversion propensity.
    """
    reg = (
        df_joined
        .select("user_id", "registered_on_date")
        .dropDuplicates(["user_id"])
        .filter(F.col("registered_on_date").isNotNull())
    )

    min_date = reg.agg(F.min("registered_on_date")).collect()[0][0]
    reg = reg.withColumn("days_since_min", F.datediff(F.col("registered_on_date"), F.lit(min_date)))
    thresholds = reg.approxQuantile("days_since_min", [0.25, 0.5, 0.75], 0.01)

    reg = reg.withColumn(
        "reg_quartile",
        F.when(F.col("days_since_min") <= thresholds[0], "Q1 (Oldest)")
         .when(F.col("days_since_min") <= thresholds[1], "Q2")
         .when(F.col("days_since_min") <= thresholds[2], "Q3")
         .otherwise("Q4 (Newest)")
    )

    opps_reg = (
        opps_enriched.select("user_id", "completed")
        .join(reg.select("user_id", "reg_quartile"), on="user_id", how="left")
    )

    agg = (
        opps_reg
        .groupBy("reg_quartile")
        .agg(F.count("*").alias("n"), F.sum("completed").alias("n_completed"))
        .withColumn("conversion_rate", F.col("n_completed") / F.col("n"))
    )

    quartile_order = ["Q1 (Oldest)", "Q2", "Q3", "Q4 (Newest)"]
    pdf = agg.toPandas()
    pdf["reg_quartile"] = pd.Categorical(pdf["reg_quartile"], categories=quartile_order, ordered=True)
    pdf = pdf.sort_values("reg_quartile")

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(pdf["reg_quartile"], pdf["conversion_rate"], color=IFOOD_RED)
    for bar, v in zip(bars, pdf["conversion_rate"]):
        if pd.notna(v):
            ax.text(bar.get_x() + bar.get_width() / 2, v,
                    f"{v:.1%}", ha="center", va="bottom", fontsize=10)
    ax.set_title("Conversion Rate by Customer Registration Quartile\n"
                 "(Q1 = earliest registered, Q4 = most recently registered)")
    ax.set_xlabel("Registration Quartile")
    ax.set_ylabel("Conversion Rate")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    plt.tight_layout()
    plt.show()
    return pdf


def plot_conversion_by_ticket_band(df_joined: DataFrame, opps_enriched: DataFrame,
                                    n_quantiles: int = 5):
    """
    Groups customers by total historical spend quintile and plots conversion rate.
    Validates the top SHAP driver: avg_ticket_before / total_spend_before.
    """
    total_spend = (
        df_joined
        .filter(F.col("event") == "transaction")
        .filter(F.col("amount") > 0)
        .groupBy("user_id")
        .agg(F.sum(F.col("amount").cast("double")).alias("total_spend"))
    )

    quantiles = [i / n_quantiles for i in range(1, n_quantiles)]
    thresholds = total_spend.approxQuantile("total_spend", quantiles, 0.02)

    def _band_expr(col_name, thresholds):
        expr = F.when(F.col(col_name).isNull(), "No history")
        for i, t in enumerate(thresholds):
            label = f"Q{i+1} (≤ {int(t):,})"
            if i == 0:
                expr = expr.when(F.col(col_name) <= t, label)
            else:
                expr = expr.when(F.col(col_name) <= t, label)
        expr = expr.otherwise(f"Q{n_quantiles} (>{int(thresholds[-1]):,})")
        return expr

    bands = total_spend.withColumn("spend_band", _band_expr("total_spend", thresholds))

    opps_with_band = (
        opps_enriched.select("user_id", "offer_id", "completed")
        .join(bands.select("user_id", "spend_band"), on="user_id", how="left")
        .withColumn("spend_band", F.coalesce(F.col("spend_band"), F.lit("No history")))
    )

    agg = (
        opps_with_band
        .groupBy("spend_band")
        .agg(
            F.count("*").alias("n_received"),
            F.sum("completed").alias("n_completed"),
        )
        .withColumn("conversion_rate", F.col("n_completed") / F.col("n_received"))
    )

    band_labels = [f"Q{i+1} (≤ {int(thresholds[i]):,})" for i in range(len(thresholds))]
    band_labels += [f"Q{n_quantiles} (>{int(thresholds[-1]):,})", "No history"]

    pdf = agg.toPandas()
    pdf["spend_band"] = pd.Categorical(pdf["spend_band"], categories=band_labels, ordered=True)
    pdf = pdf.sort_values("spend_band")

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(pdf["spend_band"], pdf["conversion_rate"], color=IFOOD_RED)
    for bar, v in zip(bars, pdf["conversion_rate"]):
        if pd.notna(v):
            ax.text(bar.get_x() + bar.get_width() / 2, v,
                    f"{v:.1%}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Historical Total Spend Quintile × Conversion Rate\n"
                 "(Validates total_spend_before — top SHAP feature)")
    ax.set_xlabel("Total Spend Band")
    ax.set_ylabel("Conversion Rate")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.show()
    return agg
