"""Load saved artifacts and score one validated farm record."""
import json
from pathlib import Path
import joblib
import pandas as pd
from src.config import DISCLAIMER, FEATURES, ROOT
from src.schemas import FarmRecord


def load_bundle(models_dir: Path = ROOT / "models") -> dict:
    """Load regressor, classifier, clusterer and metadata (clear error if not trained)."""
    meta_path = Path(models_dir) / "model_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No trained models in {models_dir}. Run: python run_all.py ...")
    return {"regressor": joblib.load(Path(models_dir) / "regressor.joblib"),
            "classifier": joblib.load(Path(models_dir) / "classifier.joblib"),
            "clusterer": joblib.load(Path(models_dir) / "clusterer.joblib"),
            "meta": json.loads(meta_path.read_text())}


def predict_record(record: dict, bundle: dict) -> dict:
    """Validate the record (raises pydantic.ValidationError) and return the prediction."""
    rec = FarmRecord.model_validate(record)
    X = pd.DataFrame([rec.model_dump()])[FEATURES]
    prob = float(bundle["classifier"].predict_proba(X)[0, 1])
    label = int(prob >= 0.5)
    return {"predicted_yield_tons": round(float(bundle["regressor"].predict(X)[0]), 4),
            "high_risk": label,
            "risk_label": "HIGH production risk - human review required" if label else "lower risk",
            "probability_high_risk": round(prob, 4),
            "cluster": int(bundle["clusterer"].predict(X)[0]),
            "verification_code": bundle["meta"]["verification_code"],
            "disclaimer": DISCLAIMER}
