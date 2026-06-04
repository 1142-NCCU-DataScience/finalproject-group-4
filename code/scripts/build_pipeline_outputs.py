#!/usr/bin/env python3
"""Regenerate batting_by_season and projection_panel from local Statcast parquet files."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from bb_pipeline.position_lahman import attach_primary_position
from bb_pipeline.projection_dataset import build_projection_panel
from bb_pipeline.statcast_season import aggregate_batting_by_season


def main() -> None:
    raw = ROOT / "data" / "raw"
    proc = ROOT / "data" / "processed"
    proc.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for y in range(2018, 2025):
        p = raw / f"statcast_{y}.parquet"
        if not p.exists():
            raise FileNotFoundError(p)
        frames.append(pd.read_parquet(p))
    sc = pd.concat(frames, ignore_index=True)

    season_df = aggregate_batting_by_season(sc)
    season_path = raw / "batting_by_season_2018_2024.parquet"
    season_df.to_parquet(season_path, index=False)
    print(f"Wrote {season_df.shape} -> {season_path}")

    season_df = attach_primary_position(season_df)
    panel = build_projection_panel(season_df, min_pa_per_feature_season=100)
    panel_path = proc / "projection_panel.parquet"
    panel.to_parquet(panel_path, index=False)
    print(f"Wrote {panel.shape} -> {panel_path}")


if __name__ == "__main__":
    main()
