"""
Unified JEPX multi-area data loader.

Returns canonical column names: DA, IM, plus exogenous regressors.
Handles datetime format differences between Tokyo and Kansai CSVs.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("/Users/aof_mac/Desktop/Full_Time_reasearcher/paper/dataset")

AREA_FILES = {
    "TK": ("IM_DA_TK_ALL.csv", "DA_TK", "IM_TK", "IMV_TK"),
    "KS": ("IM_DA_KS_ALL.csv", "DA_KS", "IM_KS", "IMV_KS"),
}

EXOG_COLS = [
    "TEMPERATURE", "RADIATION", "WIND_SPEED", "PRECIPITATION",
    "GAS_JPY", "USD_JPY", "DA_SYSTEM_PRICE",
]


def load_area(area: str) -> pd.DataFrame:
    """Load JEPX data for one area (TK or KS) with canonical column names.

    Returns a DataFrame indexed by 30-min DATETIME with columns:
        DA, IM, IMV, TEMPERATURE, RADIATION, WIND_SPEED, PRECIPITATION,
        GAS_JPY, USD_JPY, DA_SYSTEM_PRICE,
        plus calendar columns:
            hour, minute, dow, doy, month, is_weekend
    """
    if area not in AREA_FILES:
        raise ValueError(f"Unknown area {area!r}; choose from {list(AREA_FILES)}")
    fname, da_col, im_col, imv_col = AREA_FILES[area]
    df = pd.read_csv(DATA_DIR / fname)

    # Datetime: TK is ISO, KS is m/d/Y h:M (US-locale)
    df["DATETIME"] = pd.to_datetime(df["DATETIME"], errors="coerce")
    df = df.dropna(subset=["DATETIME"]).set_index("DATETIME").sort_index()

    # Canonical price columns
    df["DA"] = df[da_col].astype(float)
    df["IM"] = df[im_col].astype(float)
    df["IMV"] = df[imv_col].astype(float)

    # Calendar
    idx = df.index
    df["hour"] = idx.hour
    df["minute"] = idx.minute
    df["dow"] = idx.dayofweek
    df["doy"] = idx.dayofyear
    df["month"] = idx.month
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    # 30-min slot of the day, 0..47
    df["slot"] = df["hour"] * 2 + (df["minute"] // 30)

    keep = ["DA", "IM", "IMV"] + EXOG_COLS + [
        "hour", "minute", "dow", "doy", "month", "is_weekend", "slot",
    ]
    return df[keep]


def load_all_areas() -> dict[str, pd.DataFrame]:
    """Convenience: load both TK and KS."""
    return {a: load_area(a) for a in AREA_FILES}


if __name__ == "__main__":
    for a, d in load_all_areas().items():
        print(f"\n=== {a} ===")
        print(f"  span: {d.index.min()} -> {d.index.max()}  (n={len(d)})")
        print(f"  DA mean={d['DA'].mean():.2f} std={d['DA'].std():.2f} min={d['DA'].min():.2f} max={d['DA'].max():.2f}")
        print(f"  IM mean={d['IM'].mean():.2f} std={d['IM'].std():.2f}")
        print(f"  cols: {list(d.columns)}")
