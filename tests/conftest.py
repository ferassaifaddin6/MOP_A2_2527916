"""Shared fixtures. CHANGE these two constants to your own registration details."""
import json
import pytest
from src.config import ROOT

STUDENT_ID = "ug2527916@ines.ac.rw"
SEED = 7916

VALID = {"rainfall_mm": 1000.0, "avg_temp_c": 18.0, "soil_moisture_pct": 55.0,
         "fertilizer_kg": 120.0, "field_size_ha": 1.5, "season": "A", "cooperative_support": 1}


@pytest.fixture(scope="session")
def trained():
    """Ensure models exist (runs the pipeline once if needed)."""
    if not (ROOT / "models" / "model_meta.json").exists():
        from run_all import main
        main(STUDENT_ID, SEED)
    return json.loads((ROOT / "models" / "model_meta.json").read_text())
