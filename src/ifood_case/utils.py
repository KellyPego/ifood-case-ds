import math
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pyspark.sql import DataFrame

IFOOD_RED   = "#EA1D2C"
IFOOD_BLACK = "#3F3E3E"


def plot_financial_uplift(
    y_pred_proba: np.ndarray,
    avg_conversion_value: float,
    cost_per_contact: float,
    optimal_threshold: float,
) -> None:
    """
    Visual comparison of expected profit: model-targeted campaign vs. send-to-all baseline.

    Shows a side-by-side bar chart with:
    - Baseline profit (send to everyone)
    - Model profit (send only to score >= optimal_threshold)
    - Financial uplift (delta)
    - Customers excluded and cost saved

    Parameters
    ----------
    y_pred_proba        : predicted probabilities (1D array, positive class)
    avg_conversion_value: expected value per conversion (e.g. avg_ticket_before)
    cost_per_contact    : cost per offer sent
    optimal_threshold   : profit-maximizing threshold (selected on calibration set)
    """
    proba = np.asarray(y_pred_proba).flatten()

    baseline_profit = float(np.sum(proba * avg_conversion_value - cost_per_contact))
    mask            = proba >= optimal_threshold
    model_profit    = float(np.sum(proba[mask] * avg_conversion_value - cost_per_contact))
    uplift          = model_profit - baseline_profit
    uplift_pct      = (uplift / abs(baseline_profit)) * 100 if baseline_profit != 0 else 0.0
    targeted_n      = int(mask.sum())
    excluded_n      = len(proba) - targeted_n
    cost_saved      = excluded_n * cost_per_contact

    # --- Summary table ---
    summary = pd.DataFrame([
        {"": "Baseline (send to all)",        "Customers": f"{len(proba):,}",  "Expected Profit (BRL)": f"{baseline_profit:,.2f}"},
        {"": f"Model (thr ≥ {optimal_threshold:.2f})", "Customers": f"{targeted_n:,}",  "Expected Profit (BRL)": f"{model_profit:,.2f}"},
        {"": "Financial Uplift",              "Customers": f"−{excluded_n:,} excluded", "Expected Profit (BRL)": f"+{uplift:,.2f} (+{uplift_pct:.1f}%)"},
        {"": "Cost Saved by Exclusion",       "Customers": "",                 "Expected Profit (BRL)": f"BRL {cost_saved:,.2f}"},
    ]).set_index("")

    display(summary)

    # --- Bar chart ---
    fig, ax = plt.subplots(figsize=(8, 5))
    labels  = [f"Baseline\n(send to all)", f"Model\n(thr ≥ {optimal_threshold:.2f})"]
    values  = [baseline_profit, model_profit]
    colors  = [IFOOD_BLACK, IFOOD_RED]

    bars = ax.bar(labels, values, color=colors, width=0.4, edgecolor="white")

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.01,
            f"BRL {val:,.0f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax.annotate(
        f"Uplift: +BRL {uplift:,.0f}\n(+{uplift_pct:.1f}%)",
        xy=(1, model_profit), xytext=(1.3, (baseline_profit + model_profit) / 2),
        arrowprops=dict(arrowstyle="->", color=IFOOD_RED),
        fontsize=10, color=IFOOD_RED, fontweight="bold",
    )

    ax.set_ylabel("Expected Profit (BRL)", fontsize=11)
    ax.set_title("Financial Uplift: Model vs. Send-to-All Baseline", fontsize=13)
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0, max(values) * 1.2)
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(
    df: DataFrame,
    numerical_cols: List[str],
    target_col: str = "target",
    max_rows: int = 200000,
) -> None:
    """
    Plot correlation matrix using Pandas (no Spark VectorAssembler).

    Why this approach?
    - Avoids row loss from assembler
    - Keeps full ABT distribution
    - Produces annotated heatmap (like winning case)
    - Statistically correct for feature screening

    Parameters
    ----------
    df : Spark DataFrame (ABT)
    numerical_cols : list of numeric feature names
    target_col : name of target column
    max_rows : safety limit when converting to pandas
    """

    cols = [c for c in numerical_cols if c in df.columns]
    if target_col in df.columns:
        cols.append(target_col)

    if len(cols) == 0:
        raise ValueError("No valid numerical columns found for correlation.")

    # Optional safety cap (your ABT ~76k, so safe)
    if df.count() > max_rows:
        print(f"Dataset too large, sampling {max_rows} rows for correlation.")
        pdf = df.select(cols).limit(max_rows).toPandas()
    else:
        pdf = df.select(cols).toPandas()

    # Compute correlation matrix
    corr = pdf.corr(numeric_only=True)

    # Plot
    plt.figure(figsize=(18, 14))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        square=False,
    )
    plt.title("Correlation Matrix (Numerical Features)", fontsize=18)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_feature_distributions(
    df: DataFrame,
    numerical_cols: List[str],
    target_col: str = "target",
    top_n: int = 12,
    max_rows: int = 200_000,
) -> None:
    """
    Box plots of numerical features grouped by target (converters vs non-converters).

    Visually identifies which features have distinct distributions between the two classes,
    anticipating what SHAP will confirm quantitatively.

    Parameters
    ----------
    df           : Spark DataFrame (ABT)
    numerical_cols : list of numeric feature names
    target_col   : binary target column (0/1)
    top_n        : number of features to display (first top_n from numerical_cols)
    max_rows     : safety cap for toPandas conversion
    """
    IFOOD_RED = "#EA1D2C"

    cols = [c for c in numerical_cols if c in df.columns][:top_n]
    if not cols:
        raise ValueError("No valid numerical columns found.")

    select_cols = cols + ([target_col] if target_col in df.columns else [])
    pdf = df.select(select_cols).limit(max_rows).toPandas()

    n_cols = 3
    n_rows = math.ceil(len(cols) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        ax = axes[i]
        if target_col in pdf.columns:
            groups = pdf.groupby(target_col)[col]
            data = [grp.dropna().values for _, grp in groups]
            labels = [f"No convert (0)", "Convert (1)"]
            colors = [IFOOD_RED if lbl == "Convert (1)" else "#CCCCCC" for lbl in labels]
            bps = ax.boxplot(data, patch_artist=True, widths=0.5)
            for patch, color in zip(bps["boxes"], colors):
                patch.set_facecolor(color)
            ax.set_xticks([1, 2])
            ax.set_xticklabels(labels, fontsize=8)
        else:
            ax.boxplot(pdf[col].dropna().values, patch_artist=True)
        ax.set_title(col, fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Feature Distributions by Target (Converters vs Non-Converters)", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()


def display_policy_report(policy) -> pd.DataFrame:
    """Render a PolicyReport as a clean vertical table."""
    return pd.DataFrame({
    "Metric": [
        "Opportunities Targeted",
        "Conversions Captured",
        "Precision",
        "Recall",
        "Lift",
        "Expected Profit (BRL)",
        "Expected ROI",
        "Spend (BRL)",
    ],
    "Value": [
        f"{policy.targeted_n:,} ({policy.targeted_rate:.1%})",
        f"{policy.actual_conversions_in_target:,}",
        f"{policy.precision_in_target:.1%}",
        f"{policy.recall_in_target:.1%}",
        f"{policy.lift_in_target:.2f}×",
        f"{policy.expected_profit:,.0f}",
        f"{policy.expected_roi:.2f}×",
        f"{policy.spend:,.0f}",
    ]
})
 