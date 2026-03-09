"""
Model training module:
- Abstract base class ModelTrainer
- Concrete implementation: LGBMTrainer

Orchestrates:
1) Last-event split per customer (test = last offer per customer)
2) Group-safe split for calibration (no customer overlap)
3) Pipeline preprocessing (OHE categorical, passthrough numeric, remainder drop)
4) Hyperparameter tuning (OptunaSearchCV + GroupKFold)
5) Fit best model
6) Isotonic calibration
7) Evaluation helpers focused on ranking + ROI
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from optuna.distributions import FloatDistribution, IntDistribution
from optuna.integration import OptunaSearchCV

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from sklearn.compose import ColumnTransformer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    log_loss,
    brier_score_loss,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass(frozen=True)
class EvaluationReport:
    pr_auc: float
    roc_auc: float
    logloss: float
    brier: float
    precision_at_k: float
    recall_at_k: float
    lift_at_k: float
    k: int
    base_rate: float


class ModelTrainer(ABC):
    """
    Business rationale baked into the design:

    - Last-offer split per customer:
      simulates "train on past behavior -> predict next offer outcome" and avoids
      having the same customer offer history leaking into test via random split.

    - Group-aware CV (GroupKFold):
      reduces identity leakage (same customer appearing in multiple CV folds),
      common in event-level ABTs.

    - Calibration (isotonic):
      improves probability reliability; crucial when you use scores to estimate ROI
      and choose thresholds based on expected profit.
    """

    def __init__(
        self,
        df: DataFrame,
        numerical_columns: List[str],
        categorical_columns: List[str],
        target: str,
        id_col: str = "customer_id",
        time_col: str = "time_received",
        random_state: int = 42,
    ):
        self._df = df
        self._numerical_columns = numerical_columns
        self._categorical_columns = categorical_columns
        self._target = target
        self._id_col = id_col
        self._time_col = time_col
        self._random_state = random_state

        self._best_params: Optional[Dict[str, float]] = None
        self._best_score: Optional[float] = None
        self._estimator: Optional[Pipeline] = None
        self._calibrator: Optional[IsotonicRegression] = None

        self._x_calib: Optional[pd.DataFrame] = None
        self._y_calib: Optional[pd.Series] = None

    @property
    def best_params(self) -> Dict[str, float]:
        if self._best_params is None:
            raise RuntimeError("Model not tuned yet. Call train() first.")
        return self._best_params

    @property
    def best_score(self) -> float:
        if self._best_score is None:
            raise RuntimeError("Model not tuned yet. Call train() first.")
        return self._best_score

    @property
    def estimator(self) -> Pipeline:
        if self._estimator is None:
            raise RuntimeError("Model not trained yet. Call train() first.")
        return self._estimator

    @property
    def calibration_set(self) -> Tuple[pd.DataFrame, pd.Series]:
        if self._x_calib is None or self._y_calib is None:
            raise RuntimeError("Calibration set not available. Call train() first.")
        return self._x_calib, self._y_calib

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self._estimator is None or self._calibrator is None:
            raise RuntimeError("Model or calibrator not trained. Call train() first.")

        uncalibrated = self._estimator.predict_proba(df)[:, 1]
        calibrated = self._calibrator.predict(uncalibrated)
        calibrated = np.clip(calibrated, 0.0, 1.0)
        return np.vstack([1 - calibrated, calibrated]).T

    def predict(self, df: pd.DataFrame, threshold: float) -> np.ndarray:
        proba_pos = self.predict_proba(df)[:, 1]
        return (proba_pos >= threshold).astype(int)

    def train(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Returns x_train, x_test, y_train, y_test so your notebook can:
          - evaluate on test
          - generate curves, tables, and business insights
        """
        x_train, x_test, x_calib, y_train, y_test, y_calib, groups_train = self._split_data()

        pipeline = self._create_pipeline()
        optuna_search = self._tune_model(pipeline=pipeline, x_train=x_train, y_train=y_train, groups=groups_train)

        self._best_params = optuna_search.best_params_
        self._best_score = float(optuna_search.best_score_)
        self._estimator = optuna_search.best_estimator_

        # Store calib set for later threshold/ROI selection
        self._x_calib = x_calib
        self._y_calib = y_calib

        # Isotonic calibration
        uncalibrated_probs = self._estimator.predict_proba(x_calib)[:, 1]
        self._calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        self._calibrator.fit(uncalibrated_probs, y_calib)
        return x_train, x_test, y_train, y_test

    def evaluate_ranking(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        k: int = 5000,
    ) -> EvaluationReport:
        """
        Evaluates both:
        - probabilistic quality (PR-AUC, ROC-AUC, LogLoss, Brier)
        - ranking quality at top-K (Precision@K, Recall@K, Lift@K)

        This matches the business reality: you rarely target everyone.
        """
        proba = self.predict_proba(x)[:, 1]
        y_true = y.astype(int).values
        base_rate = float(np.mean(y_true))

        pr_auc = float(average_precision_score(y_true, proba))
        roc_auc = float(roc_auc_score(y_true, proba)) if len(np.unique(y_true)) > 1 else float("nan")
        ll = float(log_loss(y_true, np.vstack([1 - proba, proba]).T, labels=[0, 1]))
        brier = float(brier_score_loss(y_true, proba))

        k = int(min(k, len(proba)))
        order = np.argsort(-proba)  # descending
        topk = order[:k]
        topk_positives = int(np.sum(y_true[topk]))

        precision_at_k = float(topk_positives / k) if k > 0 else 0.0
        recall_at_k = float(topk_positives / max(int(np.sum(y_true)), 1))
        lift_at_k = float(precision_at_k / base_rate) if base_rate > 0 else float("nan")

        return EvaluationReport(
            pr_auc=pr_auc,
            roc_auc=roc_auc,
            logloss=ll,
            brier=brier,
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            lift_at_k=lift_at_k,
            k=k,
            base_rate=base_rate,
        )

    def find_threshold_max_profit(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        profit_if_positive: float,
        cost_if_targeted: float,
        thresholds: Optional[np.ndarray] = None,
    ) -> Tuple[float, float]:
        """
        Selects threshold to maximize expected profit.

        expected_profit_per_user = p(conversion) * profit_if_positive - cost_if_targeted
        Target users where expected profit is positive -> equivalent to threshold = cost/profit
        but we allow grid search to account for calibration imperfections.

        Returns: (best_threshold, best_total_profit)
        """
        proba = self.predict_proba(x)[:, 1]
        y_true = y.astype(int).values

        if thresholds is None:
            thresholds = np.linspace(0.01, 0.99, 99)

        best_thr = 0.5
        best_profit = -np.inf

        for thr in thresholds:
            selected = proba >= thr
            # expected profit using probabilities (not realized labels)
            exp_profit = np.sum(proba[selected] * profit_if_positive - cost_if_targeted)
            if exp_profit > best_profit:
                best_profit = float(exp_profit)
                best_thr = float(thr)

        return best_thr, best_profit

    def _split_data(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, np.ndarray]:
        """
        Split strategy:
        - Test = last offer per customer (time_received max)
        - Train = all previous offers
        - Calibration = group-split inside train (no customer overlap)

        Business impact:
        - Mimics next-offer prediction, closer to real deployment
        - Avoids optimistic metrics from random row split
        """
        w = Window.partitionBy(self._id_col)
        df_max = self._df.withColumn("max_time", F.max(F.col(self._time_col)).over(w))

        test_spark = df_max.filter(F.col(self._time_col) == F.col("max_time")).drop("max_time")
        train_spark = df_max.filter(F.col(self._time_col) < F.col("max_time")).drop("max_time")

        cols_needed = [self._id_col, self._target] + self._numerical_columns + self._categorical_columns
        train_pd = train_spark.select(*cols_needed).toPandas()
        test_pd = test_spark.select(*cols_needed).toPandas()

        x_train_full = train_pd[self._numerical_columns + self._categorical_columns]
        y_train_full = train_pd[self._target].astype(int)
        groups_full = train_pd[self._id_col].values

        x_test = test_pd[self._numerical_columns + self._categorical_columns]
        y_test = test_pd[self._target].astype(int)

        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=self._random_state)
        train_idx, calib_idx = next(gss.split(x_train_full, y_train_full, groups=groups_full))

        x_train = x_train_full.iloc[train_idx].reset_index(drop=True)
        y_train = y_train_full.iloc[train_idx].reset_index(drop=True)
        groups_train = groups_full[train_idx]

        x_calib = x_train_full.iloc[calib_idx].reset_index(drop=True)
        y_calib = y_train_full.iloc[calib_idx].reset_index(drop=True)

        return x_train, x_test, x_calib, y_train, y_test, y_calib, groups_train

    def _create_pipeline(self) -> Pipeline:
        """
        Preprocess:
        - numerical: passthrough (trees don't need scaling)
        - categorical: one-hot encode
        - remainder: drop (prevents accidental leakage)
        """
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", "passthrough", self._numerical_columns),
                ("cat", OneHotEncoder(handle_unknown="ignore"), self._categorical_columns),
            ],
            remainder="drop",
        )
        return Pipeline(steps=[("preprocess", preprocessor)])

    @abstractmethod
    def _tune_model(self, pipeline: Pipeline, x_train: pd.DataFrame, y_train: pd.Series, groups: np.ndarray) -> OptunaSearchCV:
        ...


class LGBMTrainer(ModelTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model = lgb.LGBMClassifier(
            random_state=self._random_state,
            n_jobs=-1,
            verbose=-1,
        )

    def _create_pipeline(self) -> Pipeline:
        pipeline = super()._create_pipeline()
        pipeline.steps.append(("classifier", self._model))
        return pipeline

    def _tune_model(self, pipeline: Pipeline, x_train: pd.DataFrame, y_train: pd.Series, groups: np.ndarray) -> OptunaSearchCV:
        """
        Uses GroupKFold to reduce customer leakage in CV.
        Metric: average_precision (PR-AUC), robust for imbalanced outcomes.
        """
        param_distributions = {
            "classifier__n_estimators": IntDistribution(200, 1200),
            "classifier__learning_rate": FloatDistribution(0.01, 0.2),
            "classifier__num_leaves": IntDistribution(16, 256),
            "classifier__max_depth": IntDistribution(3, 12),
            "classifier__min_child_samples": IntDistribution(10, 200),
            "classifier__subsample": FloatDistribution(0.6, 1.0),
            "classifier__colsample_bytree": FloatDistribution(0.6, 1.0),
            "classifier__reg_alpha": FloatDistribution(0.0, 2.0),
            "classifier__reg_lambda": FloatDistribution(0.0, 2.0),
        }

        cv = GroupKFold(n_splits=5)

        optuna_search = OptunaSearchCV(
            estimator=pipeline,
            param_distributions=param_distributions,
            n_trials=50,
            cv=cv,
            scoring="average_precision",
            random_state=self._random_state,
            verbose=0,
        )

        optuna_search.fit(x_train, y_train, groups=groups)
        return optuna_search