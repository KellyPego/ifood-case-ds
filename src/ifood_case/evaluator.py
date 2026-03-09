"""
Evaluator module for binary propensity models with ROI-oriented decision policies.

Includes:
- Model quality (threshold-independent): PR-AUC, ROC-AUC, LogLoss, Brier
- Diagnostics: PR/ROC curves, Confusion Matrix plot, KS statistic + plot, Calibration curve
- Ranking/business: Precision@K, Recall@K, Lift@K tables, Gains/Lift curves
- Decision policies:
  * Top-K by budget (e.g., budget=50k, cost=1 -> K=50k)
  * Threshold selected by max expected profit (choose on calibration set)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

IFOOD_RED = "#EA1D2C"
IFOOD_BLACK = "#3F3E3E"


@dataclass(frozen=True)
class ModelQualityReport:
    pr_auc: float
    roc_auc: float
    logloss: float
    brier: float
    base_rate: float


@dataclass(frozen=True)
class PolicyReport:
    policy_name: str
    targeted_n: int
    targeted_rate: float
    actual_conversions_in_target: int
    precision_in_target: float
    recall_in_target: float
    lift_in_target: float
    expected_profit: float
    expected_roi: float  # expected_profit / spend
    spend: float


class Evaluator:
    def __init__(self, y_true: Union[pd.Series, np.ndarray, List[int]]):
        y = pd.Series(y_true).astype(int).reset_index(drop=True)
        self._y_true = y

    # -------------------------
    # Helpers
    # -------------------------
    @staticmethod
    def proba_pos(y_pred_proba: np.ndarray) -> np.ndarray:
        """
        Accepts either:
          - shape (n, 2) -> uses [:, 1] as positive class probability
          - shape (n,) -> already positive class probability
        """
        if isinstance(y_pred_proba, list):
            y_pred_proba = np.asarray(y_pred_proba)

        if y_pred_proba.ndim == 1:
            return y_pred_proba.astype(float)

        if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] >= 2:
            return y_pred_proba[:, 1].astype(float)

        raise ValueError("y_pred_proba must be (n,) or (n,2+) with positive class in column 1.")

    @property
    def base_rate(self) -> float:
        return float(self._y_true.mean())

    # -------------------------
    # Model quality (threshold-independent)
    # -------------------------
    def model_quality(self, y_pred_proba: np.ndarray) -> ModelQualityReport:
        proba = self.proba_pos(y_pred_proba)
        y = self._y_true.values

        pr_auc = float(average_precision_score(y, proba))
        roc_auc = float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else float("nan")
        ll = float(log_loss(y, np.vstack([1 - proba, proba]).T, labels=[0, 1]))
        brier = float(brier_score_loss(y, proba))

        return ModelQualityReport(
            pr_auc=pr_auc,
            roc_auc=roc_auc,
            logloss=ll,
            brier=brier,
            base_rate=float(np.mean(y)),
        )

    # -------------------------
    # Curves
    # -------------------------
    def plot_pr_curve(self, y_pred_proba: np.ndarray, title: str = "Precision-Recall Curve") -> None:
        proba = self.proba_pos(y_pred_proba)
        precision, recall, _ = precision_recall_curve(self._y_true, proba)
        baseline = self.base_rate

        auc = float(average_precision_score(self._y_true, proba))
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color=IFOOD_RED, lw=2, label=f"PR Curve (AUC = {auc:.4f})")
        plt.plot([0, 1], [baseline, baseline], color=IFOOD_BLACK, lw=2, linestyle="--", label=f"Baseline ({baseline:.2f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(title)
        plt.grid(alpha=0.25)
        plt.legend(loc="best")
        plt.show()

    def plot_roc_curve(self, y_pred_proba: np.ndarray, title: str = "ROC Curve") -> None:
        proba = self.proba_pos(y_pred_proba)
        fpr, tpr, _ = roc_curve(self._y_true, proba)

        auc = float(roc_auc_score(self._y_true, proba))
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color=IFOOD_RED, lw=2, label=f"ROC Curve (AUC = {auc:.4f})")
        plt.plot([0, 1], [0, 1], color=IFOOD_BLACK, lw=2, linestyle="--", label="Random (0.50)")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(title)
        plt.grid(alpha=0.25)
        plt.legend(loc="lower right")
        plt.show()

    def plot_calibration_curve(
        self,
        y_pred_proba: np.ndarray,
        n_bins: int = 10,
        strategy: str = "quantile",
        title: str = "Calibration Curve",
    ) -> None:
        """
        Shows reliability of probabilities. Crucial for ROI decisions.
        strategy: 'uniform' or 'quantile'
        """
        proba = self.proba_pos(y_pred_proba)
        y = self._y_true.values

        frac_pos, mean_pred = calibration_curve(y, proba, n_bins=n_bins, strategy=strategy)

        plt.figure(figsize=(10, 8))
        plt.plot(mean_pred, frac_pos, marker="o", lw=2, label="Model")
        plt.plot([0, 1], [0, 1], linestyle="--", lw=2, label="Perfectly calibrated")
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Fraction of positives")
        plt.title(title)
        plt.grid(alpha=0.25)
        plt.legend(loc="best")
        plt.show()

    # -------------------------
    # Confusion + explicit conversion metrics at threshold
    # -------------------------
    def confusion_at_threshold(self, y_pred_proba: np.ndarray, threshold: float) -> np.ndarray:
        proba = self.proba_pos(y_pred_proba)
        y_pred = (proba >= threshold).astype(int)
        return confusion_matrix(self._y_true, y_pred)

    def plot_confusion_matrix(self, y_pred_proba: np.ndarray, threshold: float, title: str = "Confusion Matrix") -> None:
        cm = self.confusion_at_threshold(y_pred_proba, threshold)
        group_names = ["TN", "FP", "FN", "TP"]
        group_counts = [f"{v:0.0f}" for v in cm.flatten()]
        group_perc = [f"{v:.2%}" for v in (cm.flatten() / np.sum(cm))]
        labels = np.asarray([f"{n}\n{c}\n{p}" for n, c, p in zip(group_names, group_counts, group_perc)]).reshape(2, 2)

        cmap = sns.light_palette(IFOOD_RED, as_cmap=True)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=labels, fmt="", cmap=cmap, cbar=False)
        plt.title(f"{title} (thr={threshold:.2f})")
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.show()

    def conversion_metrics_at_threshold(self, y_pred_proba: np.ndarray, threshold: float) -> Dict[str, float]:
        proba = self.proba_pos(y_pred_proba)
        y = self._y_true.values
        y_pred = (proba >= threshold).astype(int)

        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.flatten()

        targeted_n = int(y_pred.sum())
        targeted_rate = float(targeted_n / len(y_pred)) if len(y_pred) else 0.0

        precision = float(tp / max(tp + fp, 1))
        recall = float(tp / max(tp + fn, 1))
        base_rate = float(np.mean(y))
        lift = float(precision / base_rate) if base_rate > 0 else float("nan")

        return {
            "threshold": float(threshold),
            "base_rate": base_rate,
            "targeted_n": targeted_n,
            "targeted_rate": targeted_rate,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "precision": precision,
            "recall": recall,
            "lift": lift,
        }

    # -------------------------
    # KS statistic + plot
    # -------------------------
    def ks_statistic(self, y_pred_proba: np.ndarray) -> Tuple[float, float]:
        proba = self.proba_pos(y_pred_proba)
        y = self._y_true.values

        order = np.argsort(-proba)
        y_sorted = y[order]
        p_sorted = proba[order]

        positives = (y_sorted == 1).astype(int)
        negatives = (y_sorted == 0).astype(int)

        cum_pos = np.cumsum(positives) / max(np.sum(positives), 1)
        cum_neg = np.cumsum(negatives) / max(np.sum(negatives), 1)

        diff = np.abs(cum_pos - cum_neg)
        idx = int(np.argmax(diff))

        return float(diff[idx]), float(p_sorted[idx])

    def plot_ks(self, y_pred_proba: np.ndarray, title: str = "KS Statistic") -> None:
        proba = self.proba_pos(y_pred_proba)
        y = self._y_true.values

        order = np.argsort(-proba)
        y_sorted = y[order]
        p_sorted = proba[order]

        positives = (y_sorted == 1).astype(int)
        negatives = (y_sorted == 0).astype(int)

        cum_pos = np.cumsum(positives) / max(np.sum(positives), 1)
        cum_neg = np.cumsum(negatives) / max(np.sum(negatives), 1)

        diff = np.abs(cum_pos - cum_neg)
        idx = int(np.argmax(diff))

        ks_val = float(diff[idx])
        ks_thr = float(p_sorted[idx])

        plt.figure(figsize=(10, 8))
        plt.plot(cum_pos, lw=2, label="CDF positives")
        plt.plot(cum_neg, lw=2, label="CDF negatives")
        plt.vlines(idx, ymin=min(cum_pos[idx], cum_neg[idx]), ymax=max(cum_pos[idx], cum_neg[idx]), linestyles="--", lw=2)
        plt.title(f"{title}: KS={ks_val:.3f} at thr~{ks_thr:.3f}")
        plt.xlabel("Samples sorted by score (desc)")
        plt.ylabel("Cumulative proportion")
        plt.grid(alpha=0.25)
        plt.legend(loc="best")
        plt.show()

    # -------------------------
    # Ranking @K + gains/lift curves
    # -------------------------
    def ranking_at_k(self, y_pred_proba: np.ndarray, k: int) -> Dict[str, float]:
        proba = self.proba_pos(y_pred_proba)
        y = self._y_true.values

        k = int(min(max(k, 1), len(proba)))
        order = np.argsort(-proba)
        topk = order[:k]
        topk_pos = int(np.sum(y[topk]))

        precision_k = float(topk_pos / k)
        recall_k = float(topk_pos / max(int(np.sum(y)), 1))
        lift_k = float(precision_k / self.base_rate) if self.base_rate > 0 else float("nan")

        return {
            "k": int(k),
            "k_rate": float(k / len(proba)),
            "precision_at_k": precision_k,
            "recall_at_k": recall_k,
            "lift_at_k": lift_k,
            "positives_in_topk": int(topk_pos),
        }

    def ranking_table(self, y_pred_proba: np.ndarray, ks: List[int]) -> pd.DataFrame:
        rows = [self.ranking_at_k(y_pred_proba, k) for k in ks]
        df = pd.DataFrame(rows)
        return df

    def plot_gains_lift(self, y_pred_proba: np.ndarray, title: str = "Cumulative Gains & Lift") -> None:
        """
        Gains: cumulative % of positives captured as we move down the ranked list.
        Lift: gains / random baseline.
        """
        proba = self.proba_pos(y_pred_proba)
        y = self._y_true.values

        order = np.argsort(-proba)
        y_sorted = y[order]

        total_pos = max(int(np.sum(y_sorted)), 1)
        cum_pos = np.cumsum(y_sorted) / total_pos
        population = (np.arange(len(y_sorted)) + 1) / len(y_sorted)

        # lift curve: gains / population
        lift = cum_pos / np.maximum(population, 1e-12)

        plt.figure(figsize=(10, 8))
        plt.plot(population, cum_pos, lw=2, label="Cumulative gains")
        plt.plot(population, population, lw=2, linestyle="--", label="Random baseline")
        plt.xlabel("Population targeted (fraction)")
        plt.ylabel("Positives captured (fraction)")
        plt.title(title + " - Gains")
        plt.grid(alpha=0.25)
        plt.legend(loc="best")
        plt.show()

        plt.figure(figsize=(10, 8))
        plt.plot(population, lift, lw=2, label="Lift")
        plt.hlines(1.0, 0, 1, linestyles="--", lw=2, label="Random (=1)")
        plt.xlabel("Population targeted (fraction)")
        plt.ylabel("Lift")
        plt.title(title + " - Lift")
        plt.grid(alpha=0.25)
        plt.legend(loc="best")
        plt.show()

    # -------------------------
    # Business policies: Top-K by budget and threshold by max expected profit
    # -------------------------
    @staticmethod
    def _expected_profit_from_mask(proba: np.ndarray, mask: np.ndarray, value: float, cost: float) -> float:
        selected = proba[mask]
        return float(np.sum(selected * value - cost))

    def policy_topk_by_budget(
        self,
        y_pred_proba: np.ndarray,
        budget_brl: float = 50_000.0,
        cost_per_contact: float = 1.0,
        avg_conversion_value: float = 10.0,
        policy_name: str = "Top-K by budget",
    ) -> PolicyReport:
        proba = self.proba_pos(y_pred_proba)
        y = self._y_true.values

        k = int(min(len(proba), np.floor(budget_brl / cost_per_contact)))
        if k <= 0:
            raise ValueError("Budget too small relative to cost_per_contact (K <= 0).")

        order = np.argsort(-proba)
        topk_idx = order[:k]
        mask = np.zeros(len(proba), dtype=bool)
        mask[topk_idx] = True

        y_target = y[mask]
        actual_conversions = int(np.sum(y_target))

        precision = float(actual_conversions / max(k, 1))
        recall = float(actual_conversions / max(int(np.sum(y)), 1))
        lift = float(precision / self.base_rate) if self.base_rate > 0 else float("nan")

        expected_profit = self._expected_profit_from_mask(proba, mask, avg_conversion_value, cost_per_contact)
        spend = float(k * cost_per_contact)
        expected_roi = float(expected_profit / spend) if spend > 0 else float("nan")

        return PolicyReport(
            policy_name=policy_name,
            targeted_n=k,
            targeted_rate=float(k / len(proba)),
            actual_conversions_in_target=actual_conversions,
            precision_in_target=precision,
            recall_in_target=recall,
            lift_in_target=lift,
            expected_profit=expected_profit,
            expected_roi=expected_roi,
            spend=spend,
        )

    def find_threshold_max_expected_profit(
        self,
        y_pred_proba: np.ndarray,
        avg_conversion_value: float = 10.0,
        cost_per_contact: float = 1.0,
        thresholds: Optional[np.ndarray] = None,
    ) -> Tuple[float, float]:
        proba = self.proba_pos(y_pred_proba)

        if thresholds is None:
            thresholds = np.linspace(0.01, 0.99, 99)

        best_thr = 0.5
        best_profit = -np.inf

        for thr in thresholds:
            mask = proba >= thr
            profit = self._expected_profit_from_mask(proba, mask, avg_conversion_value, cost_per_contact)
            if profit > best_profit:
                best_profit = profit
                best_thr = float(thr)

        return best_thr, float(best_profit)

    def policy_threshold(
        self,
        y_pred_proba: np.ndarray,
        threshold: float,
        avg_conversion_value: float = 10.0,
        cost_per_contact: float = 1.0,
        policy_name: str = "Threshold policy",
    ) -> PolicyReport:
        proba = self.proba_pos(y_pred_proba)
        y = self._y_true.values

        mask = proba >= threshold
        targeted_n = int(np.sum(mask))

        if targeted_n == 0:
            return PolicyReport(
                policy_name=policy_name,
                targeted_n=0,
                targeted_rate=0.0,
                actual_conversions_in_target=0,
                precision_in_target=0.0,
                recall_in_target=0.0,
                lift_in_target=float("nan"),
                expected_profit=0.0,
                expected_roi=float("nan"),
                spend=0.0,
            )

        y_target = y[mask]
        actual_conversions = int(np.sum(y_target))
        precision = float(actual_conversions / max(targeted_n, 1))
        recall = float(actual_conversions / max(int(np.sum(y)), 1))
        lift = float(precision / self.base_rate) if self.base_rate > 0 else float("nan")

        expected_profit = self._expected_profit_from_mask(proba, mask, avg_conversion_value, cost_per_contact)
        spend = float(targeted_n * cost_per_contact)
        expected_roi = float(expected_profit / spend) if spend > 0 else float("nan")

        return PolicyReport(
            policy_name=policy_name,
            targeted_n=targeted_n,
            targeted_rate=float(targeted_n / len(proba)),
            actual_conversions_in_target=actual_conversions,
            precision_in_target=precision,
            recall_in_target=recall,
            lift_in_target=lift,
            expected_profit=expected_profit,
            expected_roi=expected_roi,
            spend=spend,
        )

    # -------------------------
    # Convenience: one-call summary + report dict
    # -------------------------
    def summary(self, y_pred_proba: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
        quality = self.model_quality(y_pred_proba)
        conv = self.conversion_metrics_at_threshold(y_pred_proba, threshold)
        ks_val, ks_thr = self.ks_statistic(y_pred_proba)
        return {
            "model_quality": quality.__dict__,
            "conversion_at_threshold": conv,
            "ks": {"ks_value": ks_val, "ks_threshold": ks_thr},
        }

    def classification_report_at_threshold(self, y_pred_proba: np.ndarray, threshold: float) -> Dict[str, Any]:
        proba = self.proba_pos(y_pred_proba)
        y_pred = (proba >= threshold).astype(int)
        report_dict = classification_report(self._y_true, y_pred, output_dict=True)

        # round for readability
        for key, value in report_dict.items():
            if isinstance(value, dict):
                report_dict[key] = {k: round(v, 4) if isinstance(v, float) else v for k, v in value.items()}
            elif isinstance(value, float):
                report_dict[key] = round(value, 4)

        return report_dict

    # -------------------------
    # SHAP interpretability
    # -------------------------
    def plot_shap_summary(self, estimator: Any, x: pd.DataFrame, max_display: int = 20) -> None:
        """
        SHAP beeswarm plot using the LightGBM classifier extracted from the sklearn pipeline.

        Color = feature value (red=high, blue=low).
        X-axis = SHAP value (positive pushes toward conversion, negative away).
        Features ranked by mean |SHAP| value (global importance).

        Parameters
        ----------
        estimator : fitted sklearn Pipeline with steps "preprocess" and "classifier"
        x         : pandas DataFrame (raw features, before preprocessing)
        max_display : number of top features to show
        """
        import shap

        clf = estimator.named_steps["classifier"]
        preprocessor = estimator.named_steps["preprocess"]
        x_transformed = preprocessor.transform(x)

        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(x_transformed)

        # For binary classification shap_values is a list [class0, class1]
        vals = shap_values[1] if isinstance(shap_values, list) else shap_values

        try:
            feature_names = list(preprocessor.get_feature_names_out())
        except Exception:
            feature_names = [f"f{i}" for i in range(x_transformed.shape[1])]

        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            vals,
            x_transformed,
            feature_names=feature_names,
            max_display=max_display,
            show=False,
        )
        plt.title("SHAP Feature Impact — Propensity Model", fontsize=14)
        plt.tight_layout()
        plt.show()

    # -------------------------
    # Business: profit curve + decile table
    # -------------------------
    def plot_profit_curve(
        self,
        y_pred_proba: np.ndarray,
        avg_conversion_value: float,
        cost_per_contact: float,
        title: str = "Expected Profit by Decision Threshold",
    ) -> None:
        """
        Plots expected profit as a function of the decision threshold.

        Business rationale: visually justifies why the optimal threshold deviates from 0.5.
        A clear profit peak makes the threshold choice defensible to stakeholders.

        Expected profit at threshold t:
            sum over targeted users: [ p_i * avg_conversion_value - cost_per_contact ]
        """
        proba = self.proba_pos(y_pred_proba)
        thresholds = np.linspace(0.01, 0.99, 99)
        profits = []

        for thr in thresholds:
            mask = proba >= thr
            profit = float(np.sum(proba[mask] * avg_conversion_value - cost_per_contact)) if mask.any() else 0.0
            profits.append(profit)

        best_idx = int(np.argmax(profits))
        best_thr = float(thresholds[best_idx])
        best_profit = float(profits[best_idx])

        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, profits, color=IFOOD_RED, lw=2)
        plt.axvline(
            best_thr,
            linestyle="--",
            color=IFOOD_BLACK,
            lw=1.5,
            label=f"Optimal thr={best_thr:.2f}  |  Profit={best_profit:,.0f} BRL",
        )
        plt.axhline(0, linestyle=":", color="gray", lw=1, label="Break-even")
        plt.xlabel("Decision Threshold", fontsize=12)
        plt.ylabel("Expected Profit (BRL)", fontsize=12)
        plt.title(title, fontsize=13)
        plt.legend(fontsize=10)
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.show()

    def decile_table(self, y_pred_proba: np.ndarray) -> pd.DataFrame:
        """
        Aggregates predictions into 10 score-ranked deciles (decile 1 = highest score).

        Columns
        -------
        decile            : 1 (best) to 10 (worst)
        n                 : samples in decile
        avg_score         : mean predicted probability
        conversions       : actual positives
        conversion_rate   : conversions / n
        lift              : conversion_rate / base_rate
        cumulative_recall : fraction of all positives captured up to this decile

        Business use: tells marketing which deciles to target for a given recall
        or lift target. Standard propensity scoring deliverable.
        """
        proba = self.proba_pos(y_pred_proba)
        y = self._y_true.values
        base_rate = self.base_rate

        order = np.argsort(-proba)
        y_sorted = y[order]
        p_sorted = proba[order]

        n = len(y_sorted)
        decile_size = n // 10
        rows = []
        cum_pos = 0
        total_pos = max(int(np.sum(y)), 1)

        for d in range(10):
            start = d * decile_size
            end = (d + 1) * decile_size if d < 9 else n
            chunk_y = y_sorted[start:end]
            chunk_p = p_sorted[start:end]
            conversions = int(np.sum(chunk_y))
            cum_pos += conversions
            rate = float(conversions / max(len(chunk_y), 1))
            rows.append({
                "decile": d + 1,
                "n": len(chunk_y),
                "avg_score": round(float(np.mean(chunk_p)), 4),
                "conversions": conversions,
                "conversion_rate": round(rate, 4),
                "lift": round(float(rate / base_rate), 4) if base_rate > 0 else float("nan"),
                "cumulative_recall": round(float(cum_pos / total_pos), 4),
            })

        return pd.DataFrame(rows)