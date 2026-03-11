from __future__ import annotations

import sys
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from psyche_agent_dx.api import diagnose, health
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


if __name__ == "__main__":
    unittest.main()
