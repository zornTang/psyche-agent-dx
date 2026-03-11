# Psyche Agent Dx

A multi-agent collaborative large-model system for mental health diagnosis support.

## Vision

Psyche Agent Dx orchestrates specialized AI agents to support clinicians with:
- structured psychological intake
- risk screening and escalation
- differential diagnosis suggestions
- explainable evidence synthesis

This project is for **clinical decision support**, not a standalone medical diagnosis tool.

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

## Repository Layout

- `src/psyche_agent_dx`: core package
- `docs/architecture.md`: system design draft
