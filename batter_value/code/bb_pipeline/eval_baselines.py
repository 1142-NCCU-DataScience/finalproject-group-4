"""Baseline comparisons, error stratification, walk-forward, and classification metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from bb_pipeline.train_projection import (
    CAT_FEATURES,
    NUM_FEATURES,
    build_rf_pipeline,
    fit_by_target_season,
    prepare_xy,
    walk_forward_metrics,
)


@dataclass(frozen=True)
class ModelResult:
    """Scores for one model on one split."""

    name: str
    split: str
    mae: float
    rmse: float
    r2: float
    n_samples: int


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _metrics(name: str, split: str, y_true: np.ndarray, y_pred: np.ndarray) -> ModelResult:
    mse = mean_squared_error(y_true, y_pred)
    return ModelResult(
        name=name,
        split=split,
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(np.sqrt(mse)),
        r2=float(r2_score(y_true, y_pred)),
        n_samples=len(y_true),
    )


def attach_lag_wrc(
    panel: pd.DataFrame,
    season_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add wRC+ for lag seasons (S-3, S-2, S-1) for naive baselines.

    Uses columns from season_df: batter, season, wRC+.
    """
    out = panel.copy()
    wrc = season_df[["batter", "season", "wRC+"]].rename(columns={"season": "yr", "wRC+": "wrc"})

    for lag, off in [(3, 3), (2, 2), (1, 1)]:
        key = out[["batter", "target_season"]].copy()
        key["yr"] = key["target_season"] - off
        merged = key.merge(wrc, on=["batter", "yr"], how="left")
        out[f"wRCplus_lag{lag}"] = merged["wrc"].values

    return out


def _build_ridge_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_FEATURES),
            ("cat", _one_hot_encoder(), CAT_FEATURES),
        ]
    )
    return Pipeline([("prep", pre), ("model", Ridge(alpha=10.0, random_state=42))])


def _naive_predictions(part: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "聯盟平均 (100)": np.full(len(part), 100.0),
        "去年 wRC+ (lag1)": part["wRCplus_lag1"].values.astype(float),
        "三年 wRC+ 平均": part[["wRCplus_lag3", "wRCplus_lag2", "wRCplus_lag1"]]
        .mean(axis=1)
        .values.astype(float),
    }


def regression_baselines(
    panel: pd.DataFrame,
    *,
    train_targets: tuple[int, ...] = (2021, 2022),
    val_target: int = 2023,
    test_target: int = 2024,
) -> pd.DataFrame:
    """Compare naive baselines, Ridge, and Random Forest on validation and test."""
    tr = panel[panel["target_season"].isin(train_targets)]
    va = panel[panel["target_season"] == val_target]
    te = panel[panel["target_season"] == test_target]

    rows: list[ModelResult] = []

    for split_name, part in [("validation", va), ("test", te)]:
        y = part["wRCplus_target"].values.astype(float)
        for model_name, pred in _naive_predictions(part).items():
            pred = np.asarray(pred, dtype=float)
            if model_name == "聯盟平均 (100)":
                rows.append(_metrics(model_name, split_name, y, pred))
                continue
            mask = np.isfinite(pred)
            if mask.sum() == 0:
                continue
            rows.append(_metrics(model_name, split_name, y[mask], pred[mask]))

    x_tr, y_tr = prepare_xy(tr)
    x_va, y_va = prepare_xy(va)
    x_te, y_te = prepare_xy(te)

    ridge = _build_ridge_pipeline()
    ridge.fit(x_tr, y_tr)
    rf = build_rf_pipeline()
    rf.fit(x_tr, y_tr)

    for split_name, x_s, y_s, model_name, model in [
        ("validation", x_va, y_va, "Ridge 迴歸", ridge),
        ("test", x_te, y_te, "Ridge 迴歸", ridge),
        ("validation", x_va, y_va, "Random Forest", rf),
        ("test", x_te, y_te, "Random Forest", rf),
    ]:
        rows.append(_metrics(model_name, split_name, y_s, model.predict(x_s)))

    return pd.DataFrame([r.__dict__ for r in rows])


def error_by_tier(
    panel: pd.DataFrame,
    y_pred: np.ndarray,
    *,
    test_target: int = 2024,
    bins: tuple[float, ...] = (-np.inf, 80, 100, 130, np.inf),
    labels: tuple[str, ...] = ("<80 弱打", "80–100", "100–130", "130+ 明星"),
) -> pd.DataFrame:
    """MAE and bias by actual wRC+ tier on hold-out test."""
    te = panel[panel["target_season"] == test_target].copy().reset_index(drop=True)
    if len(y_pred) != len(te):
        raise ValueError(f"pred length {len(y_pred)} != test rows {len(te)}")

    te["pred"] = y_pred
    te["abs_err"] = (te["pred"] - te["wRCplus_target"]).abs()
    te["tier"] = pd.cut(te["wRCplus_target"], bins=bins, labels=labels)

    rows: list[dict[str, Any]] = []
    for tier, grp in te.groupby("tier", observed=True):
        rows.append(
            {
                "tier": str(tier),
                "n": len(grp),
                "mae": float(grp["abs_err"].mean()),
                "mean_actual": float(grp["wRCplus_target"].mean()),
                "mean_pred": float(grp["pred"].mean()),
                "bias": float((grp["pred"] - grp["wRCplus_target"]).mean()),
            }
        )
    return pd.DataFrame(rows)


def classification_metrics(
    panel: pd.DataFrame,
    *,
    train_targets: tuple[int, ...] = (2021, 2022),
    test_target: int = 2024,
    threshold: float = 100.0,
) -> pd.DataFrame:
    """Predict whether next-season wRC+ >= threshold (league-average hitter)."""
    tr = panel[panel["target_season"].isin(train_targets)]
    te = panel[panel["target_season"] == test_target]

    y_tr = (tr["wRCplus_target"].values >= threshold).astype(int)
    y_te = (te["wRCplus_target"].values >= threshold).astype(int)

    x_tr, _ = prepare_xy(tr)
    x_te, _ = prepare_xy(te)

    ohe = _one_hot_encoder()
    logit = Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    transformers=[
                        ("num", StandardScaler(), NUM_FEATURES),
                        ("cat", ohe, CAT_FEATURES),
                    ]
                ),
            ),
            (
                "model",
                LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced"),
            ),
        ]
    )
    logit.fit(x_tr, y_tr)

    rf_clf = Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    transformers=[
                        ("num", "passthrough", NUM_FEATURES),
                        ("cat", _one_hot_encoder(), CAT_FEATURES),
                    ]
                ),
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=12,
                    min_samples_leaf=8,
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )
    rf_clf.fit(x_tr, y_tr)

    naive_prob = np.where(
        np.isfinite(te["wRCplus_lag1"].values),
        (te["wRCplus_lag1"].values >= threshold).astype(float),
        0.5,
    )

    rows: list[dict[str, Any]] = []

    def _row(name: str, y_true: np.ndarray, prob: np.ndarray) -> dict[str, Any]:
        pred = (prob >= 0.5).astype(int)
        auc = roc_auc_score(y_true, prob) if len(np.unique(y_true)) > 1 else float("nan")
        return {
            "model": name,
            "split": "test",
            "target_season": test_target,
            "n": len(y_true),
            "accuracy": float(accuracy_score(y_true, pred)),
            "roc_auc": float(auc),
            "brier": float(brier_score_loss(y_true, np.clip(prob, 1e-6, 1 - 1e-6))),
            "positive_rate": float(y_true.mean()),
        }

    rows.append(_row("去年是否 >=100", y_te, naive_prob))
    rows.append(_row("Logistic (lag features)", y_te, logit.predict_proba(x_te)[:, 1]))
    rows.append(_row("Random Forest classifier", y_te, rf_clf.predict_proba(x_te)[:, 1]))

    return pd.DataFrame(rows)


def run_full_evaluation(
    panel: pd.DataFrame,
    season_df: pd.DataFrame,
) -> dict[str, Any]:
    """Run all evaluation pieces."""
    panel_w = attach_lag_wrc(panel, season_df)
    baseline_df = regression_baselines(panel_w)
    wf_df = walk_forward_metrics(panel_w)

    tr = panel_w[panel_w["target_season"].isin((2021, 2022))]
    te = panel_w[panel_w["target_season"] == 2024]
    x_tr, y_tr = prepare_xy(tr)
    x_te, y_te = prepare_xy(te)
    rf = build_rf_pipeline()
    rf.fit(x_tr, y_tr)
    pred_te = rf.predict(x_te)
    tier_df = error_by_tier(panel_w, pred_te, test_target=2024)
    clf_df = classification_metrics(panel_w)
    rf_main = fit_by_target_season(panel_w)

    return {
        "panel": panel_w,
        "baseline_comparison": baseline_df,
        "walk_forward": wf_df,
        "error_by_tier": tier_df,
        "classification": clf_df,
        "rf_results": rf_main,
        "rf_test_pred": pred_te,
        "rf_test_actual": y_te,
    }
