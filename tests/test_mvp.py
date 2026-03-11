from __future__ import annotations

import sys
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from psyche_agent_dx.agents import LLMDiagnosticExpertAgent
from psyche_agent_dx.api import diagnose, health
from psyche_agent_dx.config import Settings
from psyche_agent_dx.llm import ChatGeneration, ChatGLMChatModel, build_chat_model
from psyche_agent_dx.pipeline import build_default_pipeline
from psyche_agent_dx.schemas import DiagnosisRequest, RiskLevel


class PipelineSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = build_default_pipeline()

    def test_pipeline_generates_ranked_differential(self) -> None:
        response = self.pipeline.run(
            DiagnosisRequest(
                patient_text=(
                    "I have felt anxious, exhausted, and unable to sleep for weeks after "
                    "losing my job. Work and relationships are getting worse."
                ),
                clinician_context="Adult outpatient intake",
            )
        )

        self.assertEqual(len(response.planner_trace), 5)
        self.assertTrue(all(step.status == "completed" for step in response.planner_trace))
        self.assertEqual(response.report.risk_assessment.risk_level, RiskLevel.LOW)
        self.assertGreaterEqual(len(response.report.evidence), 1)
        self.assertEqual(
            response.report.differential_diagnoses[0].label,
            "Generalized anxiety disorder",
        )

    def test_pipeline_escalates_crisis_language(self) -> None:
        response = self.pipeline.run(
            DiagnosisRequest(
                patient_text=(
                    "I want to kill myself, I have a suicide plan, and I hear voices "
                    "telling me to do it."
                ),
                clinician_context="Emergency psychiatric evaluation",
            )
        )

        self.assertEqual(response.report.risk_assessment.risk_level, RiskLevel.CRISIS)
        self.assertIn(
            "Immediate human review is required",
            " ".join(response.report.risk_assessment.recommended_actions),
        )


class ApiHandlerSmokeTests(unittest.TestCase):
    def test_api_handlers_return_expected_payloads(self) -> None:
        health_payload = health()
        response = diagnose(
            DiagnosisRequest(
                patient_text=(
                    "I have felt anxious, exhausted, and unable to sleep for weeks after "
                    "losing my job. Work and relationships are getting worse."
                ),
                clinician_context="Adult outpatient intake",
            )
        )

        self.assertEqual(health_payload["status"], "ok")
        self.assertIn("diagnosis_backend", health_payload)
        self.assertEqual(len(response.planner_trace), 5)
        self.assertEqual(response.report.risk_assessment.risk_level, RiskLevel.LOW)
        self.assertGreaterEqual(len(response.report.differential_diagnoses), 1)


class FakeChatModel:
    def __init__(self, text: str | None = None, error: Exception | None = None) -> None:
        self._text = text
        self._error = error
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> ChatGeneration:
        self.prompts.append(prompt)
        if self._error is not None:
            raise self._error
        return ChatGeneration(text=self._text or "")


class FallbackDiagnosticAgent:
    def __init__(self) -> None:
        self.calls = 0

    def diagnose(self, structured_case, safety, evidence):
        self.calls += 1
        return [
            {
                "label": "fallback",
                "structured_case": structured_case,
                "safety": safety,
                "evidence": evidence,
            }
        ]


class ChatGLMPathTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = build_default_pipeline()
        self.request = DiagnosisRequest(
            patient_text=(
                "I have felt anxious, exhausted, and unable to sleep for weeks after "
                "losing my job. Work and relationships are getting worse."
            ),
            clinician_context="Adult outpatient intake",
        )
        structured_case = self.pipeline._intake.normalize(self.request)
        safety = self.pipeline._risk_expert.assess(structured_case)
        evidence = self.pipeline._retrieval.retrieve(structured_case, self.request.clinician_context)
        self.structured_case = structured_case
        self.safety = safety
        self.evidence = evidence

    def test_llm_agent_uses_valid_json_output(self) -> None:
        model = FakeChatModel(
            text="""
            preliminary text
            {
              "differential_diagnoses": [
                {
                  "label": "Adjustment disorder",
                  "rationale": "Linked to job loss and functional decline.",
                  "confidence": 0.61,
                  "evidence_ids": ["dsm5-adjustment"]
                },
                {
                  "label": "Generalized anxiety disorder",
                  "rationale": "Worry and sleep disturbance remain prominent.",
                  "confidence": 1.4,
                  "evidence_ids": ["dsm5-gad", "missing-id"]
                }
              ]
            }
            trailing text
            """
        )
        agent = LLMDiagnosticExpertAgent(chat_model=model)

        candidates = agent.diagnose(self.structured_case, self.safety, self.evidence)

        self.assertEqual([item.label for item in candidates], ["Generalized anxiety disorder", "Adjustment disorder"])
        self.assertEqual(candidates[0].confidence, 1.0)
        self.assertEqual(candidates[0].evidence_ids, ["dsm5-gad"])
        self.assertEqual(len(model.prompts), 1)

    def test_llm_agent_falls_back_on_invalid_json(self) -> None:
        fallback = FallbackDiagnosticAgent()
        model = FakeChatModel(text="not-json-at-all")
        agent = LLMDiagnosticExpertAgent(chat_model=model, fallback_agent=fallback)

        candidates = agent.diagnose(self.structured_case, self.safety, self.evidence)

        self.assertEqual(fallback.calls, 1)
        self.assertEqual(candidates[0]["label"], "fallback")

    def test_llm_agent_falls_back_on_model_error(self) -> None:
        fallback = FallbackDiagnosticAgent()
        model = FakeChatModel(error=RuntimeError("model crashed"))
        agent = LLMDiagnosticExpertAgent(chat_model=model, fallback_agent=fallback)

        candidates = agent.diagnose(self.structured_case, self.safety, self.evidence)

        self.assertEqual(fallback.calls, 1)
        self.assertEqual(candidates[0]["label"], "fallback")

    def test_build_chat_model_switches_on_backend(self) -> None:
        rule_model = build_chat_model(Settings(diagnosis_backend="rule"))
        chatglm_model = build_chat_model(Settings(diagnosis_backend="chatglm"))

        self.assertIsNone(rule_model)
        self.assertIsInstance(chatglm_model, ChatGLMChatModel)


if __name__ == "__main__":
    unittest.main()
