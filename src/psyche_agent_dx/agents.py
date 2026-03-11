from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import re

from psyche_agent_dx.knowledge import InMemoryKnowledgeBase
from psyche_agent_dx.llm import ChatModel
from psyche_agent_dx.prompts import build_diagnostic_prompt
from psyche_agent_dx.schemas import (
    DiagnosticCandidate,
    DiagnosticReport,
    DiagnosisRequest,
    EvidenceChunk,
    PlannerStep,
    RiskLevel,
    SafetyAssessment,
    StructuredCase,
)


@dataclass(frozen=True)
class PipelineContext:
    request: DiagnosisRequest
    structured_case: StructuredCase
    safety: SafetyAssessment
    evidence: list[EvidenceChunk]
    candidates: list[DiagnosticCandidate]


class PlannerAgent:
    def create_plan(self, request: DiagnosisRequest) -> list[PlannerStep]:
        has_context = bool(request.clinician_context)
        return [
            PlannerStep(name="intake_normalization", purpose="Normalize free-text intake into a case schema.", status="pending"),
            PlannerStep(name="risk_screening", purpose="Check for crisis and high-risk mental health signals.", status="pending"),
            PlannerStep(name="knowledge_retrieval", purpose=f"Retrieve DSM-5 / CBT / safety evidence{' with clinician context' if has_context else ''}.", status="pending"),
            PlannerStep(name="diagnostic_reasoning", purpose="Generate differential diagnoses with explicit evidence linkage.", status="pending"),
            PlannerStep(name="report_coordination", purpose="Compose a compliant clinician-facing support report.", status="pending"),
        ]


class IntakeAgent:
    _symptom_map = {
        "sleep": "sleep disturbance",
        "insomnia": "sleep disturbance",
        "sad": "low mood",
        "depressed": "low mood",
        "anxious": "anxiety",
        "anxiety": "anxiety",
        "worry": "excessive worry",
        "panic": "panic symptoms",
        "fatigue": "fatigue",
        "tired": "fatigue",
        "focus": "concentration problems",
        "hopeless": "hopelessness",
        "withdraw": "social withdrawal",
    }
    _stressor_markers = ("breakup", "job", "exam", "family", "divorce", "loss", "financial")
    _impairment_markers = ("work", "school", "relationship", "daily life", "function")
    _risk_markers = ("suicide", "kill myself", "self-harm", "hurt myself", "voices")

    def normalize(self, request: DiagnosisRequest) -> StructuredCase:
        text = " ".join(filter(None, [request.patient_text, request.clinician_context])).lower()
        symptoms = [label for token, label in self._symptom_map.items() if token in text]
        stressors = [marker for marker in self._stressor_markers if marker in text]
        impairments = [marker for marker in self._impairment_markers if marker in text]
        risks = [marker for marker in self._risk_markers if marker in text]

        summary = request.patient_text.strip()
        return StructuredCase(
            summary=summary,
            reported_symptoms=_dedupe(symptoms),
            stressors=_dedupe(stressors),
            functional_impairments=_dedupe(impairments),
            risk_flags=_dedupe(risks),
        )


class RiskExpertAgent:
    _crisis_terms = ("kill myself", "end my life", "suicide plan", "command hallucinations")
    _high_terms = ("suicidal", "self-harm", "hurt myself", "voices", "violent")
    _moderate_terms = ("hopeless", "can't cope", "drinking heavily", "severe anxiety")

    def assess(self, structured_case: StructuredCase) -> SafetyAssessment:
        text = " ".join(
            [
                structured_case.summary.lower(),
                " ".join(structured_case.risk_flags).lower(),
                " ".join(structured_case.reported_symptoms).lower(),
            ]
        )

        if any(term in text for term in self._crisis_terms):
            return SafetyAssessment(
                risk_level=RiskLevel.CRISIS,
                rationale="Detected language consistent with imminent self-harm or psychotic crisis risk.",
                recommended_actions=[
                    "Immediate human review is required before any automated advice is used.",
                    "Direct the user to emergency services or a crisis hotline based on local protocol.",
                ],
            )
        if any(term in text for term in self._high_terms):
            return SafetyAssessment(
                risk_level=RiskLevel.HIGH,
                rationale="Detected self-harm, violence, or severe perceptual-disturbance indicators.",
                recommended_actions=[
                    "Escalate to a clinician for same-day safety assessment.",
                    "Limit the system response to supportive guidance and crisis resources.",
                ],
            )
        if any(term in text for term in self._moderate_terms):
            return SafetyAssessment(
                risk_level=RiskLevel.MODERATE,
                rationale="Detected meaningful distress that warrants structured follow-up and monitoring.",
                recommended_actions=[
                    "Recommend clinician follow-up and symptom monitoring.",
                    "Provide coping guidance without presenting a definitive diagnosis.",
                ],
            )
        return SafetyAssessment(
            risk_level=RiskLevel.LOW,
            rationale="No explicit crisis language detected in the intake text.",
            recommended_actions=[
                "Continue standard intake and differential screening.",
            ],
        )


class RetrievalAgent:
    def __init__(self, knowledge_base: InMemoryKnowledgeBase) -> None:
        self._knowledge_base = knowledge_base

    def retrieve(self, structured_case: StructuredCase, clinician_context: str | None = None) -> list[EvidenceChunk]:
        query_parts = [
            structured_case.summary,
            " ".join(structured_case.reported_symptoms),
            " ".join(structured_case.stressors),
            clinician_context or "",
        ]
        return self._knowledge_base.search(" ".join(query_parts), limit=4)


class DiagnosticExpertAgent:
    def diagnose(
        self,
        structured_case: StructuredCase,
        safety: SafetyAssessment,
        evidence: list[EvidenceChunk],
    ) -> list[DiagnosticCandidate]:
        summary_text = structured_case.summary.lower()
        symptoms = set(structured_case.reported_symptoms)
        impairment_present = bool(structured_case.functional_impairments)
        candidates: list[DiagnosticCandidate] = []

        depression_score = sum(
            symptom in symptoms
            for symptom in (
                "low mood",
                "fatigue",
                "sleep disturbance",
                "concentration problems",
                "hopelessness",
                "social withdrawal",
            )
        )
        anxiety_score = sum(
            symptom in symptoms
            for symptom in (
                "anxiety",
                "excessive worry",
                "sleep disturbance",
                "panic symptoms",
            )
        )

        if depression_score >= 2 or (depression_score >= 1 and impairment_present):
            candidates.append(
                DiagnosticCandidate(
                    label="Major depressive disorder",
                    rationale="Low mood, fatigue, sleep disruption, and impairment markers align with a depressive presentation.",
                    confidence=min(0.5 + (depression_score * 0.08) + (0.05 if impairment_present else 0.0), 0.82),
                    evidence_ids=_matching_evidence_ids(evidence, "depression"),
                )
            )
        if anxiety_score >= 2 or ("anxiety" in symptoms and impairment_present):
            candidates.append(
                DiagnosticCandidate(
                    label="Generalized anxiety disorder",
                    rationale="Persistent worry and arousal-related symptoms support an anxiety-spectrum differential.",
                    confidence=min(0.48 + (anxiety_score * 0.08) + (0.04 if impairment_present else 0.0), 0.8),
                    evidence_ids=_matching_evidence_ids(evidence, "anxiety"),
                )
            )
        if structured_case.stressors and (anxiety_score >= 1 or depression_score >= 1 or impairment_present):
            candidates.append(
                DiagnosticCandidate(
                    label="Adjustment disorder",
                    rationale="Symptoms appear linked to identifiable stressors and possible functional decline.",
                    confidence=0.58 if impairment_present else 0.52,
                    evidence_ids=_matching_evidence_ids(evidence, "adjustment"),
                )
            )
        if "panic" in summary_text or "panic symptoms" in symptoms:
            candidates.append(
                DiagnosticCandidate(
                    label="Panic disorder",
                    rationale="Panic-related language suggests episodic autonomic arousal that requires clarification.",
                    confidence=0.54,
                    evidence_ids=_matching_evidence_ids(evidence, "anxiety"),
                )
            )

        if not candidates:
            candidates.append(
                DiagnosticCandidate(
                    label="Unspecified anxiety or mood-related distress",
                    rationale="The current evidence supports distress screening but is insufficient for a narrower differential.",
                    confidence=0.4 if safety.risk_level == RiskLevel.LOW else 0.3,
                    evidence_ids=[item.id for item in evidence[:2]],
                )
            )

        candidates.sort(key=lambda item: item.confidence, reverse=True)
        return candidates[:3]


class LLMDiagnosticExpertAgent:
    def __init__(self, chat_model: ChatModel, fallback_agent: DiagnosticExpertAgent | None = None) -> None:
        self._chat_model = chat_model
        self._fallback_agent = fallback_agent or DiagnosticExpertAgent()

    def diagnose(
        self,
        structured_case: StructuredCase,
        safety: SafetyAssessment,
        evidence: list[EvidenceChunk],
    ) -> list[DiagnosticCandidate]:
        prompt = build_diagnostic_prompt(structured_case, safety, evidence)

        try:
            response = self._chat_model.generate(prompt)
            return self._parse_candidates(response.text, evidence) or self._fallback_agent.diagnose(
                structured_case,
                safety,
                evidence,
            )
        except Exception:
            return self._fallback_agent.diagnose(structured_case, safety, evidence)

    def _parse_candidates(
        self,
        payload: str,
        evidence: list[EvidenceChunk],
    ) -> list[DiagnosticCandidate]:
        match = re.search(r"\{.*\}", payload, re.DOTALL)
        if not match:
            return []

        parsed = json.loads(match.group(0))
        items = parsed.get("differential_diagnoses", [])
        evidence_ids = {item.id for item in evidence}
        candidates: list[DiagnosticCandidate] = []

        for item in items[:3]:
            label = str(item.get("label", "")).strip()
            rationale = str(item.get("rationale", "")).strip()
            confidence = float(item.get("confidence", 0.0))
            raw_evidence_ids = item.get("evidence_ids", [])
            linked_evidence_ids = [
                evidence_id
                for evidence_id in raw_evidence_ids
                if isinstance(evidence_id, str) and evidence_id in evidence_ids
            ]
            if not label or not rationale:
                continue
            candidates.append(
                DiagnosticCandidate(
                    label=label,
                    rationale=rationale,
                    confidence=max(0.0, min(confidence, 1.0)),
                    evidence_ids=linked_evidence_ids,
                )
            )
        candidates.sort(key=lambda item: item.confidence, reverse=True)
        return candidates


class CoordinatorAgent:
    _notice = (
        "This system provides mental health decision support only. It does not replace "
        "licensed clinical assessment, emergency triage, or formal diagnosis."
    )

    def compose(self, context: PipelineContext) -> DiagnosticReport:
        guidance = list(context.safety.recommended_actions)
        if any(candidate.label == "Major depressive disorder" for candidate in context.candidates):
            guidance.append("Consider CBT behavioral activation and monitor changes in sleep, energy, and anhedonia.")
        if any(candidate.label == "Generalized anxiety disorder" for candidate in context.candidates):
            guidance.append("Consider CBT cognitive restructuring and worry-monitoring homework.")
        if not guidance:
            guidance.append("Collect more longitudinal symptom data before narrowing the differential.")

        return DiagnosticReport(
            generated_at=datetime.now(timezone.utc),
            case_summary=context.structured_case,
            risk_assessment=context.safety,
            differential_diagnoses=context.candidates,
            care_guidance=_dedupe(guidance),
            evidence=context.evidence,
            coordinator_notes=(
                "Planner completed intake normalization, risk screening, retrieval, and "
                "differential reasoning. Outputs should be reviewed by a clinician."
            ),
            compliance_notice=self._notice,
        )


def _matching_evidence_ids(evidence: list[EvidenceChunk], keyword: str) -> list[str]:
    keyword = keyword.lower()
    return [
        item.id
        for item in evidence
        if keyword in item.title.lower() or keyword in " ".join(item.tags).lower()
    ]


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered
