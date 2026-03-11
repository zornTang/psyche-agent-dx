# Architecture Draft

## Core Components

- Intake Agent: normalizes symptoms/history into structured fields.
- Risk Agent: checks suicide/self-harm/violence/substance risk signals.
- Diagnostic Agent: proposes differential diagnosis candidates.
- Evidence Agent: maps candidates to rationale and clinical evidence.
- Coordinator Agent: composes a final report with confidence and caveats.

## Data Flow

1. User or clinician submits intake text.
2. Intake Agent builds canonical case representation.
3. Risk Agent runs first and may trigger urgent escalation.
4. Diagnostic + Evidence Agents run in parallel.
5. Coordinator Agent merges outputs into a traceable summary.

## Safety Principles

- Human-in-the-loop for any high-risk recommendation.
- Explicit uncertainty and confidence reporting.
- Audit logs for each agent decision step.
