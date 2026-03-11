# Psyche Agent Dx

A multi-agent collaborative large-model system for mental health diagnosis support.

## Vision

Psyche Agent Dx orchestrates specialized AI agents to support clinicians with:
- structured psychological intake
- risk screening and escalation
- differential diagnosis suggestions
- explainable evidence synthesis

This project is for **clinical decision support**, not a standalone medical diagnosis tool.

## Current Scope

This repository now contains a runnable MVP skeleton with:
- `PlannerAgent` for workflow orchestration
- `RiskExpertAgent` for crisis triage
- `RetrievalAgent` for DSM-5 / CBT / safety evidence recall
- `DiagnosticExpertAgent` for differential generation
- `CoordinatorAgent` for compliant report composition
- `FastAPI` endpoints for health checks and diagnosis support

The current implementation uses a lightweight in-memory knowledge base and rule-based reasoning so the architecture can be exercised before swapping in ChatGLM2, vector retrieval, and a frontend.

## ChatGLM Integration

The backend can optionally switch from rule-based differential generation to a local ChatGLM model.

Recommended model:
- `THUDM/chatglm2-6b`

Supported fallback option:
- `THUDM/chatglm-6b`

Install optional local-LLM dependencies:

```bash
pip install -e '.[local-llm]'
```

Run with ChatGLM2:

```bash
export PSYCHE_DIAGNOSIS_BACKEND=chatglm
export PSYCHE_CHATGLM_MODEL_ID=THUDM/chatglm2-6b
export PSYCHE_CHATGLM_DEVICE=cuda
export PSYCHE_CHATGLM_QUANTIZATION_BITS=4
python -m psyche_agent_dx.main
```

Optional version pinning:

```bash
export PSYCHE_CHATGLM_REVISION=v1.0
```

The current integration uses ChatGLM only for differential generation and keeps the rule-based fallback in place. That is intentional for safety and reliability in clinical-support flows.

## Planned Multi-Agent Workflow

1. Intake Agent: gathers structured patient context.
2. Risk Agent: performs crisis and safety risk checks.
3. Diagnostic Agent: generates differential hypotheses.
4. Evidence Agent: links guidance/evidence to each hypothesis.
5. Coordinator Agent: merges outputs into one traceable report.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m psyche_agent_dx.main
```

Open `http://127.0.0.1:8000/docs` for the API schema.

Example request:

```bash
curl -X POST http://127.0.0.1:8000/diagnose \
  -H "Content-Type: application/json" \
  -d '{
    "patient_text": "I have felt anxious, exhausted, and unable to sleep for weeks after losing my job. Work and relationships are getting worse.",
    "clinician_context": "Adult outpatient intake"
  }'
```

## Repository Layout

- `src/psyche_agent_dx`: core package
- `docs/architecture.md`: system design draft

## Next Build Targets

1. Replace rule-based diagnostic reasoning with ChatGLM2 prompts and structured output parsing.
2. Replace in-memory retrieval with chunking, embeddings, hybrid search, and reranking.
3. Add audit logging, prompt templates, and policy-driven safety interception.
4. Add a web UI for intake capture, evidence viewing, and report review.
