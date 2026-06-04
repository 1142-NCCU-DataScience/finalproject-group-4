"""Train RandomForest and XGBoost regressors with season-based splits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

XGBRegressor = None  # lazy load; OpenMP may be missing on some macOS setups


def _try_xgb():
    """Return XGBRegressor class or None if import/runtime fails."""
    global XGBRegressor  # noqa: PLW0603
    if XGBRegressor is not None:
        return XGBRegressor
    try:
        from xgboost import XGBRegressor as XGBR  # noqa: WPS433

        _ = XGBR  # touch lib
        XGBRegressor = XGBR
        return XGBR
    except Exception:  # noqa: BLE001
        return None


NUM_FEATURES = [
    "K%_lag3",
    "K%_lag2",
    "K%_lag1",
    "BB%_lag3",
    "BB%_lag2",
    "BB%_lag1",
    "BABIP_lag3",
    "BABIP_lag2",
    "BABIP_lag1",
    "BIP%_lag3",
    "BIP%_lag2",
    "BIP%_lag1",
    "PA_lag3",
    "PA_lag2",
    "PA_lag1",
    "PA_sum_lag3",
    "age_t",
]

CAT_FEATURES = ["primary_pos"]


def _prep_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    use_cols = [c for c in NUM_FEATURES + CAT_FEATURES if c in df.columns]
    missing = [c for c in NUM_FEATURES + CAT_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"projection panel missing columns: {missing}")
    x = df[use_cols].copy()
    x["age_t"] = x["age_t"].fillna(x["age_t"].median())
    y = df["wRCplus_target"].values.astype(float)
    return x, y


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Public wrapper for model inputs."""
    return _prep_xy(df)


def _one_hot() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_rf_pipeline() -> Pipeline:
    """Random forest with one-hot positions."""
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUM_FEATURES),
            ("cat", _one_hot(), CAT_FEATURES),
        ]
    )
    rf = RandomForestRegressor(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=8,
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("prep", pre), ("model", rf)])


def build_xgb_pipeline() -> Pipeline:
    """XGBoost regressor."""
    XGBR = _try_xgb()
    if XGBR is None:
        raise ImportError("xgboost is not available")

    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUM_FEATURES),
            ("cat", _one_hot(), CAT_FEATURES),
        ]
    )
    xgb = XGBR(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    return Pipeline([("prep", pre), ("model", xgb)])


@dataclass
class SplitMetrics:
    """Regression scores on a subset."""

    mae: float
    rmse: float
    r2: float
    n_samples: int


def evaluate(model: Pipeline, x: pd.DataFrame, y: np.ndarray) -> SplitMetrics:
    pred = model.predict(x)
    mse = mean_squared_error(y, pred)
    rmse_v = float(np.sqrt(mse))
    return SplitMetrics(
        mae=float(mean_absolute_error(y, pred)),
        rmse=rmse_v,
        r2=float(r2_score(y, pred)),
        n_samples=len(y),
    )


def fit_by_target_season(
    panel: pd.DataFrame,
    *,
    train_targets: tuple[int, ...] = (2021, 2022),
    val_target: int = 2023,
    test_target: int = 2024,
) -> dict[str, Any]:
    """
    Train / val / test using target_season (no random shuffle).

    Returns dict with fitted pipelines, metrics, and permutation importance on validation.
    """
    tr = panel[panel["target_season"].isin(train_targets)].copy()
    va = panel[panel["target_season"] == val_target].copy()
    te = panel[panel["target_season"] == test_target].copy()

    x_tr, y_tr = _prep_xy(tr)
    x_va, y_va = _prep_xy(va)
    x_te, y_te = _prep_xy(te)

    out: dict[str, Any] = {
        "train_targets": train_targets,
        "val_target": val_target,
        "test_target": test_target,
        "n_train": len(tr),
        "n_val": len(va),
        "n_test": len(te),
    }

    rf = build_rf_pipeline()
    rf.fit(x_tr, y_tr)
    out["rf_val"] = evaluate(rf, x_va, y_va)
    out["rf_test"] = evaluate(rf, x_te, y_te)
    out["rf_model"] = rf

    pi = permutation_importance(
        rf, x_va, y_va, n_repeats=15, random_state=42, n_jobs=-1
    )
    out["rf_perm_importance_mean"] = dict(
        zip(
            _feature_names_after_prep(rf, len(pi.importances_mean)),
            pi.importances_mean.tolist(),
        )
    )

    if _try_xgb() is not None:
        xgb = build_xgb_pipeline()
        xgb.fit(x_tr, y_tr)
        out["xgb_val"] = evaluate(xgb, x_va, y_va)
        out["xgb_test"] = evaluate(xgb, x_te, y_te)
        out["xgb_model"] = xgb
        pix = permutation_importance(
            xgb, x_va, y_va, n_repeats=15, random_state=42, n_jobs=-1
        )
        out["xgb_perm_importance_mean"] = dict(
            zip(
                _feature_names_after_prep(xgb, len(pix.importances_mean)),
                pix.importances_mean.tolist(),
            )
        )

    return out


def _feature_names_after_prep(pipe: Pipeline, n_features: int) -> list[str]:
    """Names after ColumnTransformer; fallback to generic labels."""
    prep: ColumnTransformer = pipe.named_steps["prep"]
    try:
        names = list(prep.get_feature_names_out())
        if len(names) == n_features:
            return names
    except Exception:
        pass
    return [f"feature_{i}" for i in range(n_features)]


def walk_forward_metrics(panel: pd.DataFrame) -> pd.DataFrame:
    """Expanding-window train and next season as test."""
    seasons = sorted(panel["target_season"].unique())
    rows: list[dict[str, Any]] = []
    if len(seasons) < 2:
        return pd.DataFrame()

    for i in range(2, len(seasons)):
        test_s = seasons[i]
        train_targets = tuple(seasons[:i])
        tr = panel[panel["target_season"].isin(train_targets)]
        te = panel[panel["target_season"] == test_s]
        if len(tr) < 50 or len(te) < 20:
            continue
        x_tr, y_tr = _prep_xy(tr)
        x_te, y_te = _prep_xy(te)
        rf = build_rf_pipeline()
        rf.fit(x_tr, y_tr)
        m = evaluate(rf, x_te, y_te)
        rows.append(
            {
                "test_target_season": test_s,
                "mae": m.mae,
                "rmse": m.rmse,
                "r2": m.r2,
                "n_test": m.n_samples,
            }
        )

    return pd.DataFrame(rows)
