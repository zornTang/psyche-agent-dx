from __future__ import annotations

from psyche_agent_dx.agents import (
    CoordinatorAgent,
    DiagnosticExpertAgent,
    IntakeAgent,
    LLMDiagnosticExpertAgent,
    PipelineContext,
    PlannerAgent,
    RetrievalAgent,
    RiskExpertAgent,
)
from psyche_agent_dx.config import get_settings
from psyche_agent_dx.knowledge import InMemoryKnowledgeBase
from psyche_agent_dx.llm import build_chat_model
from psyche_agent_dx.schemas import DiagnosisRequest, DiagnosisResponse


class DiagnosisPipeline:
    def __init__(
        self,
        planner: PlannerAgent,
        intake: IntakeAgent,
        risk_expert: RiskExpertAgent,
        retrieval: RetrievalAgent,
        diagnostic_expert: DiagnosticExpertAgent,
        coordinator: CoordinatorAgent,
    ) -> None:
        self._planner = planner
        self._intake = intake
        self._risk_expert = risk_expert
        self._retrieval = retrieval
        self._diagnostic_expert = diagnostic_expert
        self._coordinator = coordinator

    def run(self, request: DiagnosisRequest) -> DiagnosisResponse:
        planner_trace = self._planner.create_plan(request)

        structured_case = self._intake.normalize(request)
        planner_trace[0].status = "completed"

        safety = self._risk_expert.assess(structured_case)
        planner_trace[1].status = "completed"

        evidence = self._retrieval.retrieve(structured_case, request.clinician_context)
        planner_trace[2].status = "completed"

        candidates = self._diagnostic_expert.diagnose(structured_case, safety, evidence)
        planner_trace[3].status = "completed"

        report = self._coordinator.compose(
            PipelineContext(
                request=request,
                structured_case=structured_case,
                safety=safety,
                evidence=evidence,
                candidates=candidates,
            )
        )
        planner_trace[4].status = "completed"

        return DiagnosisResponse(planner_trace=planner_trace, report=report)


def build_default_pipeline() -> DiagnosisPipeline:
    settings = get_settings()
    knowledge_base = InMemoryKnowledgeBase()
    diagnostic_expert = DiagnosticExpertAgent()
    chat_model = build_chat_model(settings)
    if chat_model is not None:
        diagnostic_expert = LLMDiagnosticExpertAgent(
            chat_model=chat_model,
            fallback_agent=diagnostic_expert,
        )

    return DiagnosisPipeline(
        planner=PlannerAgent(),
        intake=IntakeAgent(),
        risk_expert=RiskExpertAgent(),
        retrieval=RetrievalAgent(knowledge_base),
        diagnostic_expert=diagnostic_expert,
        coordinator=CoordinatorAgent(),
    )
