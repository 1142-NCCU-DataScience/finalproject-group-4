"""Build projection rows: features from seasons S-3,S-2,S-1 → target wRC+ in season S."""

from __future__ import annotations

import pandas as pd

from bb_pipeline.position_lahman import attach_primary_position

RATE_COLS = ("K%", "BB%", "BABIP", "BIP%")


def build_projection_panel(
    season_df: pd.DataFrame,
    *,
    min_pa_per_feature_season: int = 100,
    target_seasons: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    """
    Each row: predict `next_season_wRCplus` = wRC+ in season S using stats from S-3, S-2, S-1.

    Age (`age_t`) = age_bat_median in season S-1 (most recent year in the feature window).
    Position = primary fielding position in season S-1 (via Lahman; UNK if missing).

    Args:
        season_df: Output of `aggregate_batting_by_season` (with `primary_pos` optional).
        min_pa_per_feature_season: Drop feature years with PA below this.
        target_seasons: If set, only emit rows where S is in this set (e.g. (2021,2022,2023,2024)).

    Returns:
        DataFrame with lag features, `anchor_season` = S-1, `target_season` = S, and target wRC+.
    """
    df = season_df.copy()
    if "primary_pos" not in df.columns:
        df = attach_primary_position(df)

    df = df.sort_values(["batter", "season"])
    rows: list[dict] = []

    if target_seasons is None:
        target_seasons = tuple(range(int(df["season"].min()) + 3, int(df["season"].max()) + 2))

    for batter, grp in df.groupby("batter", sort=False):
        g = grp.set_index("season")
        for s in target_seasons:
            if s not in g.index:
                continue
            need = (s - 3, s - 2, s - 1)
            if any(y not in g.index for y in need):
                continue
            r3, r2, r1 = g.loc[need[0]], g.loc[need[1]], g.loc[need[2]]
            if (
                r3["PA"] < min_pa_per_feature_season
                or r2["PA"] < min_pa_per_feature_season
                or r1["PA"] < min_pa_per_feature_season
            ):
                continue

            row: dict = {
                "batter": int(batter),
                "Name": r1.get("Name", ""),
                "Team": r1.get("Team", ""),
                "target_season": int(s),
                "anchor_season": int(s - 1),
                "wRCplus_target": float(g.loc[s]["wRC+"]),
                "PA_sum_lag3": int(r3["PA"] + r2["PA"] + r1["PA"]),
                "age_t": float(r1["age_bat_median"])
                if pd.notna(r1["age_bat_median"])
                else float("nan"),
                "primary_pos": r1.get("primary_pos", "UNK"),
            }
            for col in RATE_COLS:
                row[f"{col}_lag3"] = float(r3[col])
                row[f"{col}_lag2"] = float(r2[col])
                row[f"{col}_lag1"] = float(r1[col])
            row["PA_lag3"] = int(r3["PA"])
            row["PA_lag2"] = int(r2["PA"])
            row["PA_lag1"] = int(r1["PA"])
            rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["primary_pos"] = out["primary_pos"].fillna("UNK").astype(str)
    return out.sort_values(["target_season", "wRCplus_target"], ascending=[True, False]).reset_index(
        drop=True
    )
