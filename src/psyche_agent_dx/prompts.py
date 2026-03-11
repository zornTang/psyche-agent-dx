from __future__ import annotations

import json

from psyche_agent_dx.schemas import EvidenceChunk, SafetyAssessment, StructuredCase


def build_diagnostic_prompt(
    structured_case: StructuredCase,
    safety: SafetyAssessment,
    evidence: list[EvidenceChunk],
) -> str:
    evidence_payload = [
        {
            "id": item.id,
            "title": item.title,
            "source": item.source.value,
            "content": item.content,
            "tags": item.tags,
        }
        for item in evidence
    ]

    instructions = {
        "role": "mental_health_support_agent",
        "task": "Generate a differential diagnosis shortlist for clinician decision support.",
        "constraints": [
            "Do not claim a formal diagnosis.",
            "Use retrieved evidence only when possible.",
            "If evidence is weak, lower confidence and stay broad.",
            "If risk is high or crisis, avoid extensive therapeutic advice.",
            "Return strict JSON only.",
        ],
        "output_schema": {
            "differential_diagnoses": [
                {
                    "label": "string",
                    "rationale": "string",
                    "confidence": "float between 0 and 1",
                    "evidence_ids": ["string"],
                }
            ]
        },
        "case": {
            "summary": structured_case.summary,
            "reported_symptoms": structured_case.reported_symptoms,
            "stressors": structured_case.stressors,
            "functional_impairments": structured_case.functional_impairments,
            "risk_flags": structured_case.risk_flags,
        },
        "risk_assessment": {
            "risk_level": safety.risk_level.value,
            "rationale": safety.rationale,
            "recommended_actions": safety.recommended_actions,
        },
        "evidence": evidence_payload,
    }
    return json.dumps(instructions, ensure_ascii=False, indent=2)
