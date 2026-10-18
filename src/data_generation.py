"""Reproducible synthetic dataset for the Musanze potato cooperative."""
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd


def generate_dataframe(seed: int) -> pd.DataFrame:
    """Build base rows, then inject missing values and 3 exact duplicates."""
    rng = np.random.default_rng(seed)
    n = 240 + seed % 41
    rain = rng.uniform(600, 1400, n)
    temp = rng.uniform(12, 26, n)
    soil = rng.uniform(20, 80, n)
    fert = rng.uniform(0, 250, n)
    size = rng.uniform(0.2, 4.0, n)
    season = rng.choice(["A", "B"], n)
    support = rng.integers(0, 2, n)
    noise = rng.normal(0, 0.80, n)

    season_effect = np.where(season == "A", 0.50, -0.20)
    y = (0.004 * rain + 0.06 * soil + 0.02 * fert + 1.30 * size
         - 0.25 * np.abs(temp - 18) + 0.70 * support + season_effect + noise)
    y = np.clip(y, 0.50, None)
    per_ha = y / size
    df = pd.DataFrame({"rainfall_mm": rain, "avg_temp_c": temp, "soil_moisture_pct": soil,
                       "fertilizer_kg": fert, "field_size_ha": size, "season": season,
                       "cooperative_support": support, "yield_tons": y,
                       "high_risk": (per_ha < np.median(per_ha)).astype(int)})

    m = round(0.05 * n)
    df.loc[rng.choice(n, m, replace=False), "soil_moisture_pct"] = np.nan
    df.loc[rng.choice(n, m, replace=False), "fertilizer_kg"] = np.nan
    dup_rows = df.iloc[rng.choice(n, 3, replace=False)]
    return pd.concat([df, dup_rows], ignore_index=True)


def write_dataset(seed: int, csv_path: Path, fingerprint_path: Path, code: str) -> str:
    """Write CSV (4 dp, LF endings) and its SHA-256 fingerprint; return the hash."""
    df = generate_dataframe(seed)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, float_format="%.4f", lineterminator="\n")
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    fingerprint_path.parent.mkdir(parents=True, exist_ok=True)
    fingerprint_path.write_text(f"sha256: {digest}\nfile: {csv_path.name}\nrows: {len(df)}\n"
                                f"verification_code: {code}\n")
    return digest
