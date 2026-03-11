# Architecture Draft

## MVP Goal

Build a clinician-support prototype that accepts unstructured mental-health intake text and returns:
- a normalized case summary
- a safety risk assessment
- a small differential diagnosis list
- linked DSM-5 / CBT / safety evidence
- compliant, traceable care guidance

## Runtime Components

- `PlannerAgent`
  Routes the workflow and records execution steps for auditability.
- `IntakeAgent`
  Normalizes free text into a canonical case schema.
- `RiskExpertAgent`
  Runs before other reasoning steps and can force escalation behavior.
- `RetrievalAgent`
  Pulls structured evidence from the knowledge layer.
- `DiagnosticExpertAgent`
  Generates differential diagnoses grounded in retrieved evidence.
- `CoordinatorAgent`
  Produces the final report, uncertainty notes, and compliance notice.

## Data Contracts

- `DiagnosisRequest`
  API input with `patient_text` and optional `clinician_context`.
- `StructuredCase`
  Normalized symptoms, stressors, functional impairments, and risk flags.
- `SafetyAssessment`
  Risk level, rationale, and recommended actions.
- `EvidenceChunk`
  Retrieved knowledge snippet with `source`, `tags`, and retrieval `score`.
- `DiagnosticCandidate`
  Differential item with confidence and linked evidence ids.
- `DiagnosticReport`
  Final coordinator output returned to the frontend or clinician workflow.

## Current MVP Flow

1. Client submits intake text to `/diagnose`.
2. `PlannerAgent` emits a deterministic plan trace.
3. `IntakeAgent` extracts core symptoms, stressors, and impairment markers.
4. `RiskExpertAgent` screens for crisis indicators.
5. `RetrievalAgent` queries the knowledge base for DSM-5, CBT, and safety snippets.
6. `DiagnosticExpertAgent` creates ranked differentials.
7. `CoordinatorAgent` assembles a compliant response payload.

## Target v1 Evolution

1. Replace the rule-based intake and diagnosis logic with ChatGLM2 structured prompts.
2. Add hybrid retrieval:
   - dense embeddings for semantic recall
   - BM25 for lexical recall
   - reranker for evidence ordering
3. Add explicit policy modules:
   - crisis interception
   - refusal / downgrade behavior
   - protected-topic compliance templates
4. Add observability:
   - prompt and response logging
   - evidence provenance
   - latency and failure metrics
5. Add delivery surfaces:
   - FastAPI backend
   - Web intake and evidence-review UI

## Suggested Directory Growth

- `src/psyche_agent_dx/agents.py`
  Current MVP agents. Split into a package once prompts and tools grow.
- `src/psyche_agent_dx/knowledge.py`
  Current in-memory corpus. Replace with loaders, chunkers, and retrieval adapters.
- `src/psyche_agent_dx/pipeline.py`
  Workflow orchestration entry point.
- `src/psyche_agent_dx/api.py`
  FastAPI surface for frontend and integration use.

## Safety Principles

- Human review is mandatory for any high-risk or crisis-classified case.
- Reports must state uncertainty and avoid claiming formal diagnosis.
- Retrieved evidence should be cited in the output whenever clinical suggestions are made.
- Internal reasoning can be structured and logged, but user-facing responses should remain concise and compliant.
