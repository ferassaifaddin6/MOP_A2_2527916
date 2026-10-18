"""Typed, strict request/response models shared by the CLI and the API."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field
from src.config import RANGES


class FarmRecord(BaseModel):
    """One farm record. Strict types: '800' (a string) is rejected, unknown fields are rejected."""
    model_config = ConfigDict(strict=True, extra="forbid")
    rainfall_mm: float = Field(ge=RANGES["rainfall_mm"][0], le=RANGES["rainfall_mm"][1])
    avg_temp_c: float = Field(ge=RANGES["avg_temp_c"][0], le=RANGES["avg_temp_c"][1])
    soil_moisture_pct: float = Field(ge=RANGES["soil_moisture_pct"][0], le=RANGES["soil_moisture_pct"][1])
    fertilizer_kg: float = Field(ge=RANGES["fertilizer_kg"][0], le=RANGES["fertilizer_kg"][1])
    field_size_ha: float = Field(ge=RANGES["field_size_ha"][0], le=RANGES["field_size_ha"][1])
    season: Literal["A", "B"]
    cooperative_support: Literal[0, 1]


class PredictionResponse(BaseModel):
    """Prediction returned by predict.py and POST /predict."""
    predicted_yield_tons: float
    high_risk: int
    risk_label: str
    probability_high_risk: float
    cluster: int
    verification_code: str
    disclaimer: str
