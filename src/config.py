"""Central configuration: paths, column groups and valid ranges."""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

NUMERIC = ["rainfall_mm", "avg_temp_c", "soil_moisture_pct", "fertilizer_kg", "field_size_ha"]
CATEGORICAL = ["season"]
BINARY = ["cooperative_support"]
FEATURES = NUMERIC + CATEGORICAL + BINARY
TARGET_REG = "yield_tons"
TARGET_CLS = "high_risk"
REQUIRED_COLUMNS = FEATURES + [TARGET_REG, TARGET_CLS]
FORBIDDEN_FEATURES = {"yield_tons", "yield_per_ha", "high_risk"}

RANGES = {
    "rainfall_mm": (600.0, 1400.0),
    "avg_temp_c": (12.0, 26.0),
    "soil_moisture_pct": (20.0, 80.0),
    "fertilizer_kg": (0.0, 250.0),
    "field_size_ha": (0.2, 4.0),
}
TEST_SIZE = 0.2
K_RANGE = range(2, 7)
DISCLAIMER = ("Classroom decision-support prototype on SYNTHETIC data. Not a replacement "
              "for an agronomist; must not be used for finance, insurance or public-service decisions.")


@dataclass
class Paths:
    """All output locations for one student."""
    root: Path
    regdigits: str

    @property
    def csv(self): return self.root / "data" / f"AI_A2_{self.regdigits}.csv"
    @property
    def models(self): return self.root / "models"
    @property
    def artifacts(self): return self.root / "artifacts"
    @property
    def plots(self): return self.artifacts / "plots"
    @property
    def fingerprint(self): return self.artifacts / "data_fingerprint.txt"

    def ensure(self):
        for p in (self.csv.parent, self.models, self.plots):
            p.mkdir(parents=True, exist_ok=True)
