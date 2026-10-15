"""Pydantic schemas for the Medical Triage API."""

from pydantic import BaseModel, Field


class TriageRequest(BaseModel):
    """Input payload containing the medical abstract/report text."""

    medical_abstract: str = Field(
        ...,
        min_length=10,
        description="Texto do laudo ou resumo clínico para classificação e triagem.",
        examples=[
            "The patient presented with acute ST-elevation myocardial infarction "
            "and severe left ventricular dysfunction requiring immediate catheterization."
        ],
    )


class TriageResponse(BaseModel):
    """Output payload returned by the inference endpoint."""

    condition_label: int
    condition_name: str
    urgency: str
    latency_ms: float
    model_backend: str
