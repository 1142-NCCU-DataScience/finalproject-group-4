"""Aggregate Statcast pitch-level data to batter-season batting lines."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pybaseball import playerid_reverse_lookup

PA_EVENTS = frozenset(
    {
        "single",
        "double",
        "triple",
        "home_run",
        "walk",
        "intent_walk",
        "hit_by_pitch",
        "strikeout",
        "strikeout_double_play",
        "field_out",
        "force_out",
        "grounded_into_double_play",
        "double_play",
        "fielders_choice",
        "fielders_choice_out",
        "field_error",
        "sac_fly",
        "sac_fly_double_play",
        "sac_bunt",
        "sac_bunt_double_play",
        "catcher_interf",
    }
)

# Balls-in-play ending events ( fair contact / balls put in play excluding walks, K, HBP; excludes sac bunt).
BIP_EVENTS = frozenset(
    {
        "single",
        "double",
        "triple",
        "home_run",
        "field_out",
        "force_out",
        "grounded_into_double_play",
        "double_play",
        "fielders_choice",
        "fielders_choice_out",
        "field_error",
        "sac_fly",
        "sac_fly_double_play",
    }
)

WOBA_SCALE = 1.18
LG_R_PA = 0.127


def _batter_season_rows(sc_pa: pd.DataFrame) -> list[dict[str, Any]]:
    """One dict per (batter, season) from PA-ending Statcast rows."""
    if "game_year" not in sc_pa.columns:
        raise ValueError("Statcast data must include game_year")

    sc_pa = sc_pa.copy()
    sc_pa["season"] = sc_pa["game_year"].astype(int)

    rows: list[dict[str, Any]] = []
    for (batter_id, season), grp in sc_pa.groupby(["batter", "season"], sort=False):
        ev = grp["events"]
        pa = len(ev)
        if pa == 0:
            continue

        h1 = (ev == "single").sum()
        h2 = (ev == "double").sum()
        h3 = (ev == "triple").sum()
        hr = (ev == "home_run").sum()
        h = int(h1 + h2 + h3 + hr)
        bb = ev.isin(["walk", "intent_walk"]).sum()
        hbp = (ev == "hit_by_pitch").sum()
        sf = ev.isin(["sac_fly", "sac_fly_double_play"]).sum()
        sb = ev.isin(["sac_bunt", "sac_bunt_double_play"]).sum()
        k = ev.isin(["strikeout", "strikeout_double_play"]).sum()
        ab = pa - bb - hbp - sf - sb
        avg = h / ab if ab > 0 else 0.0
        denom_obp = ab + bb + hbp + sf
        obp = (h + bb + hbp) / denom_obp if denom_obp > 0 else 0.0
        slg = (h1 + 2 * h2 + 3 * h3 + 4 * hr) / ab if ab > 0 else 0.0
        iso = slg - avg
        babip_denom = ab - k - hr + sf
        babip = (h - hr) / babip_denom if babip_denom > 0 else 0.0

        wd = grp["woba_value"].sum(skipna=True)
        wdn = grp["woba_denom"].sum(skipna=True)
        woba = float(wd / wdn) if wdn > 0 else 0.0

        bip_ct = int(ev.isin(BIP_EVENTS).sum())
        bip_pct = bip_ct / pa if pa else 0.0

        age_med = np.nan
        if "age_bat" in grp.columns:
            age_med = float(grp["age_bat"].median())

        g = int(grp["game_pk"].nunique()) if "game_pk" in grp.columns else 0

        rows.append(
            {
                "batter": int(batter_id),
                "season": int(season),
                "G": g,
                "PA": int(pa),
                "AB": int(ab),
                "H": h,
                "2B": int(h2),
                "3B": int(h3),
                "HR": int(hr),
                "BB": int(bb),
                "K": int(k),
                "HBP": int(hbp),
                "SF": int(sf),
                "AVG": round(avg, 3),
                "OBP": round(obp, 3),
                "SLG": round(slg, 3),
                "OPS": round(obp + slg, 3),
                "ISO": round(iso, 3),
                "BABIP": round(babip, 3),
                "BB%": round(bb / pa, 4),
                "K%": round(k / pa, 4),
                "BIP%": round(bip_pct, 4),
                "wOBA": round(woba, 3),
                "age_bat_median": age_med,
            }
        )

    return rows


def _apply_season_wrc_war(df: pd.DataFrame) -> pd.DataFrame:
    """League wOBA-weighted wRC+ and simplified WAR per season (same formula as notebook)."""
    out = df.copy()
    out["wRC+"] = 100.0
    out["WAR"] = 0.0

    for season in sorted(out["season"].unique()):
        mask = out["season"] == season
        sub = out.loc[mask]
        pa_sum = sub["PA"].sum()
        if pa_sum <= 0:
            continue
        lg_woba = float((sub["wOBA"] * sub["PA"]).sum() / pa_sum)

        wrc = []
        war = []
        for _, r in sub.iterrows():
            if r["PA"] > 0:
                wrc.append(
                    round(
                        ((r["wOBA"] - lg_woba) / WOBA_SCALE / LG_R_PA + 1) * 100,
                        1,
                    )
                )
                war.append(
                    round(
                        (
                            (r["wOBA"] - lg_woba) / WOBA_SCALE * r["PA"]
                            + 20 * (r["PA"] / 600)
                        )
                        / 10,
                        1,
                    )
                )
            else:
                wrc.append(100.0)
                war.append(0.0)
        out.loc[mask, "wRC+"] = wrc
        out.loc[mask, "WAR"] = war

    return out


def _attach_names_and_team(df: pd.DataFrame, sc_pa: pd.DataFrame) -> pd.DataFrame:
    """Merge player names and modal home_team per batter-season."""
    ids = [str(x) for x in df["batter"].astype(int).unique().tolist()]
    ndf = playerid_reverse_lookup(ids, key_type="mlbam")
    ndf["Name"] = ndf["name_first"].str.title() + " " + ndf["name_last"].str.title()
    ndf = ndf.rename(columns={"key_mlbam": "batter"})
    ndf["batter"] = ndf["batter"].astype(int)
    df = df.merge(ndf[["batter", "Name"]], on="batter", how="left")
    df["Name"] = df["Name"].fillna("Unknown_" + df["batter"].astype(str))

    sc_pa = sc_pa[sc_pa["events"].isin(PA_EVENTS)].copy()
    sc_pa["season"] = sc_pa["game_year"].astype(int)
    tmap = (
        sc_pa.groupby(["batter", "season"])["home_team"]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
    )
    tmap.columns = ["batter", "season", "Team"]
    df = df.merge(tmap, on=["batter", "season"], how="left")
    df["Team"] = df["Team"].fillna("UNK")
    return df


def aggregate_batting_by_season(sc_full: pd.DataFrame) -> pd.DataFrame:
    """
    Build batter-season totals from pitch-level Statcast.

    BIP% = (# PA-ending rows whose event is in BIP_EVENTS) / PA.

    Args:
        sc_full: Statcast rows (one or many seasons); must include events, game_year,
            batter, woba_value, woba_denom, home_team, game_pk, age_bat.

    Returns:
        DataFrame with one row per (batter, season).
    """
    sc_pa = sc_full[sc_full["events"].isin(PA_EVENTS)].copy()
    rows = _batter_season_rows(sc_pa)
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = _apply_season_wrc_war(df)
    df = _attach_names_and_team(df, sc_full)

    col_order = [
        "Name",
        "Team",
        "batter",
        "season",
        "G",
        "PA",
        "AB",
        "H",
        "2B",
        "3B",
        "HR",
        "BB",
        "K",
        "HBP",
        "SF",
        "AVG",
        "OBP",
        "SLG",
        "OPS",
        "ISO",
        "BABIP",
        "BB%",
        "K%",
        "BIP%",
        "wOBA",
        "wRC+",
        "WAR",
        "age_bat_median",
    ]
    return df[[c for c in col_order if c in df.columns]].sort_values(
        ["season", "wRC+"], ascending=[True, False]
    ).reset_index(drop=True)
