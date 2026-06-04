"""Primary defensive position from Lahman / Chadwick (not Statcast)."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _lahman_fielding_long() -> Optional[pd.DataFrame]:
    """playerID, yearID, POS with games G (best-effort Lahman download)."""
    try:
        import pybaseball.lahman as lahman

        lahman.download_lahman()
        fld = lahman.fielding()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Lahman fielding unavailable: %s", exc)
        return None

    need = {"playerID", "yearID", "POS", "G"}
    if not need.issubset(set(fld.columns)):
        logger.warning("Lahman Fielding.csv missing expected columns")
        return None

    agg = fld.groupby(["playerID", "yearID", "POS"], as_index=False)["G"].sum()
    # Primary = position with most games played that season
    idx = agg.groupby(["playerID", "yearID"])["G"].idxmax()
    prim = agg.loc[idx].rename(columns={"POS": "primary_pos_lahman"})
    return prim[["playerID", "yearID", "primary_pos_lahman"]]


@lru_cache(maxsize=1)
def _mlbam_to_player_id() -> Optional[pd.DataFrame]:
    """Map MLBAM batter id -> Lahman playerID via Chadwick key_bbref."""
    try:
        from pybaseball import chadwick_register
        import pybaseball.lahman as lahman

        lahman.download_lahman()
        people = lahman.people()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Lahman people unavailable: %s", exc)
        return None

    cr = chadwick_register()
    cr = cr.dropna(subset=["key_mlbam"])
    cr["key_mlbam"] = cr["key_mlbam"].astype(int)
    cr = cr[cr["key_mlbam"] > 0]

    bbref_col = "bbrefID" if "bbrefID" in people.columns else "bbref_id"
    if bbref_col not in people.columns:
        logger.warning("Lahman People missing bbref id column")
        return None

    m = cr.merge(
        people[["playerID", bbref_col]].drop_duplicates(subset=[bbref_col]),
        left_on="key_bbref",
        right_on=bbref_col,
        how="inner",
    )
    out = m[["key_mlbam", "playerID"]].drop_duplicates(subset=["key_mlbam"])
    out = out.rename(columns={"key_mlbam": "batter"})
    return out


def attach_primary_position(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add column `primary_pos` per (batter, season).

    Args:
        df: Must include `batter` (MLBAM int) and `season`.

    Returns:
        Copy with `primary_pos` (str); `'UNK'` when unknown.
    """
    out = df.copy()
    fld = _lahman_fielding_long()
    pid_map = _mlbam_to_player_id()

    if fld is None or pid_map is None:
        out["primary_pos"] = "UNK"
        return out

    x = out.merge(pid_map, on="batter", how="left")
    x = x.merge(
        fld,
        left_on=["playerID", "season"],
        right_on=["playerID", "yearID"],
        how="left",
    )
    x["primary_pos"] = x["primary_pos_lahman"].fillna("UNK")
    drop_cols = [c for c in ("playerID", "yearID", "primary_pos_lahman") if c in x.columns]
    x = x.drop(columns=drop_cols, errors="ignore")
    return x
