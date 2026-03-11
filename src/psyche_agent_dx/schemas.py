from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRISIS = "crisis"


class EvidenceSource(str, Enum):
    DSM5 = "dsm5"
    CBT = "cbt"
    SAFETY = "safety"


class DiagnosisRequest(BaseModel):
    patient_text: str = Field(..., min_length=10, description="Free-form intake text")
    clinician_context: str | None = Field(
        default=None,
        description="Optional context from clinician or intake form",
    )


class StructuredCase(BaseModel):
    summary: str
    reported_symptoms: list[str] = Field(default_factory=list)
    stressors: list[str] = Field(default_factory=list)
    functional_impairments: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)


class EvidenceChunk(BaseModel):
    id: str
    title: str
    source: EvidenceSource
    content: str
    tags: list[str] = Field(default_factory=list)
    score: float


class SafetyAssessment(BaseModel):
    risk_level: RiskLevel
    rationale: str
    recommended_actions: list[str] = Field(default_factory=list)


class DiagnosticCandidate(BaseModel):
    label: str
    rationale: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence_ids: list[str] = Field(default_factory=list)


class DiagnosticReport(BaseModel):
    generated_at: datetime
    case_summary: StructuredCase
    risk_assessment: SafetyAssessment
    differential_diagnoses: list[DiagnosticCandidate] = Field(default_factory=list)
    care_guidance: list[str] = Field(default_factory=list)
    evidence: list[EvidenceChunk] = Field(default_factory=list)
    coordinator_notes: str
    compliance_notice: str


class PlannerStep(BaseModel):
    name: str
    purpose: str
    status: str


class DiagnosisResponse(BaseModel):
    planner_trace: list[PlannerStep]
    report: DiagnosticReport
