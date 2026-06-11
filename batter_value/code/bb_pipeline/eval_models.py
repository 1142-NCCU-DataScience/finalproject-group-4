"""Supervised model comparison (no rule-based naive baselines)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from bb_pipeline.eval_baselines import ModelResult, _metrics, attach_lag_wrc
from bb_pipeline.train_projection import (
    CAT_FEATURES,
    NUM_FEATURES,
    build_rf_pipeline,
    prepare_xy,
)


WRC_NUM = ["wRCplus_lag3", "wRCplus_lag2", "wRCplus_lag1", "PA_sum_lag3", "age_t"]
WRC_ALL = WRC_NUM + ["primary_pos"]


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _prep_process_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    return prepare_xy(df)


def _prep_wrc_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Learned model on historical wRC+ trajectory (not fixed average)."""
    use = [c for c in WRC_ALL if c in df.columns]
    x = df[use].copy()
    x["age_t"] = x["age_t"].fillna(x["age_t"].median())
    for c in WRC_NUM:
        if c in x.columns:
            x[c] = x[c].fillna(x[c].median())
    y = df["wRCplus_target"].values.astype(float)
    return x, y


COMBINED_NUM = NUM_FEATURES + ["wRCplus_lag3", "wRCplus_lag2", "wRCplus_lag1"]


def _prep_combined_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Process rates + historical wRC+ lags in one supervised model."""
    use = [c for c in COMBINED_NUM + CAT_FEATURES if c in df.columns]
    x = df[use].copy()
    x["age_t"] = x["age_t"].fillna(x["age_t"].median())
    for c in ["wRCplus_lag3", "wRCplus_lag2", "wRCplus_lag1"]:
        if c in x.columns:
            x[c] = x[c].fillna(x[c].median())
    y = df["wRCplus_target"].values.astype(float)
    return x, y


def _scaled_combined_prep() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), COMBINED_NUM),
            ("cat", _one_hot_encoder(), CAT_FEATURES),
        ]
    )


def _scaled_process_prep() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_FEATURES),
            ("cat", _one_hot_encoder(), CAT_FEATURES),
        ]
    )


def _scaled_wrc_prep() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), WRC_NUM),
            ("cat", _one_hot_encoder(), CAT_FEATURES),
        ]
    )


def _passthrough_process_prep() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUM_FEATURES),
            ("cat", _one_hot_encoder(), CAT_FEATURES),
        ]
    )


def _try_xgb_pipeline() -> Pipeline | None:
    try:
        from xgboost import XGBRegressor  # noqa: WPS433

        pre = _passthrough_process_prep()
        model = XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        return Pipeline([("prep", pre), ("model", model)])
    except Exception:
        return None


def get_model_registry() -> list[tuple[str, str, Callable[[], Pipeline], Callable[[pd.DataFrame], tuple[pd.DataFrame, np.ndarray]]]]:
    """
    Return (display name, family, factory, prep_fn).

    family: 'process' | 'wrc_history' | 'combined'
    """
    registry: list[tuple[str, str, Callable[[], Pipeline], Any]] = [
        (
            "Ridge — 過程指標",
            "process",
            lambda: Pipeline(
                [("prep", _scaled_process_prep()), ("model", Ridge(alpha=10.0, random_state=42))]
            ),
            _prep_process_xy,
        ),
        (
            "ElasticNet — 過程指標",
            "process",
            lambda: Pipeline(
                [
                    ("prep", _scaled_process_prep()),
                    ("model", ElasticNet(alpha=0.05, l1_ratio=0.5, random_state=42, max_iter=5000)),
                ]
            ),
            _prep_process_xy,
        ),
        (
            "Random Forest — 過程指標",
            "process",
            build_rf_pipeline,
            _prep_process_xy,
        ),
        (
            "HistGradientBoosting — 過程指標",
            "process",
            lambda: Pipeline(
                [
                    ("prep", _passthrough_process_prep()),
                    (
                        "model",
                        HistGradientBoostingRegressor(
                            max_depth=8,
                            learning_rate=0.05,
                            max_iter=400,
                            random_state=42,
                        ),
                    ),
                ]
            ),
            _prep_process_xy,
        ),
        (
            "Ridge — 歷年 wRC+ 軌跡",
            "wrc_history",
            lambda: Pipeline(
                [("prep", _scaled_wrc_prep()), ("model", Ridge(alpha=5.0, random_state=42))]
            ),
            _prep_wrc_xy,
        ),
        (
            "ElasticNet — 歷年 wRC+ 軌跡",
            "wrc_history",
            lambda: Pipeline(
                [
                    ("prep", _scaled_wrc_prep()),
                    ("model", ElasticNet(alpha=0.03, l1_ratio=0.3, random_state=42, max_iter=5000)),
                ]
            ),
            _prep_wrc_xy,
        ),
        (
            "Ridge — 過程指標 + 歷年 wRC+",
            "combined",
            lambda: Pipeline(
                [
                    ("prep", _scaled_combined_prep()),
                    ("model", Ridge(alpha=10.0, random_state=42)),
                ]
            ),
            _prep_combined_xy,
        ),
    ]
    xgb = _try_xgb_pipeline()
    if xgb is not None:
        registry.append(
            ("XGBoost — 過程指標", "process", lambda: xgb, _prep_process_xy)  # noqa: E731
        )
    return registry


def regression_models_comparison(
    panel: pd.DataFrame,
    *,
    train_targets: tuple[int, ...] = (2021, 2022),
    val_target: int = 2023,
    test_target: int = 2024,
) -> pd.DataFrame:
    """Fit all supervised models; return metrics on validation and test."""
    tr = panel[panel["target_season"].isin(train_targets)]
    va = panel[panel["target_season"] == val_target]
    te = panel[panel["target_season"] == test_target]

    rows: list[ModelResult] = []

    for name, _family, factory, prep_fn in get_model_registry():
        pipe = factory()
        x_tr, y_tr = prep_fn(tr)
        x_va, y_va = prep_fn(va)
        x_te, y_te = prep_fn(te)
        pipe.fit(x_tr, y_tr)
        for split_name, x_s, y_s in [
            ("validation", x_va, y_va),
            ("test", x_te, y_te),
        ]:
            rows.append(_metrics(name, split_name, y_s, pipe.predict(x_s)))

    out = pd.DataFrame([r.__dict__ for r in rows])
    out["family"] = out["name"].map(
        {n: fam for n, fam, _, _ in get_model_registry()}
    )
    return out


def best_model_on_test(comparison_df: pd.DataFrame) -> str:
    test = comparison_df[comparison_df["split"] == "test"].sort_values("mae")
    if test.empty:
        return "Random Forest — 過程指標"
    return str(test.iloc[0]["name"])


def fit_model_by_name(panel: pd.DataFrame, model_name: str) -> tuple[Pipeline, Callable[[pd.DataFrame], tuple[pd.DataFrame, np.ndarray]]]:
    for name, _, factory, prep_fn in get_model_registry():
        if name == model_name:
            pipe = factory()
            tr = panel[panel["target_season"].isin((2021, 2022))]
            x_tr, y_tr = prep_fn(tr)
            pipe.fit(x_tr, y_tr)
            return pipe, prep_fn
    raise ValueError(f"Unknown model: {model_name}")


def run_model_evaluation(
    panel: pd.DataFrame,
    season_df: pd.DataFrame,
) -> dict[str, Any]:
    """Full supervised-only evaluation."""
    panel_w = attach_lag_wrc(panel, season_df)
    models_df = regression_models_comparison(panel_w)
    best_name = best_model_on_test(models_df)
    pipe, prep_fn = fit_model_by_name(panel_w, best_name)

    te = panel_w[panel_w["target_season"] == 2024]
    x_te, y_te = prep_fn(te)
    pred_te = pipe.predict(x_te)

    from bb_pipeline.eval_baselines import error_by_tier

    tier_df = error_by_tier(panel_w, pred_te, test_target=2024)

    return {
        "panel": panel_w,
        "models_comparison": models_df,
        "best_model_name": best_name,
        "best_test_pred": pred_te,
        "best_test_actual": y_te,
        "error_by_tier": tier_df,
    }
