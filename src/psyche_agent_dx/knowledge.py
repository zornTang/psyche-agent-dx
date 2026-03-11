from __future__ import annotations

import re
from dataclasses import dataclass

from psyche_agent_dx.schemas import EvidenceChunk, EvidenceSource


TOKEN_PATTERN = re.compile(r"[a-zA-Z]{3,}")


@dataclass(frozen=True)
class KnowledgeDocument:
    id: str
    title: str
    source: EvidenceSource
    content: str
    tags: tuple[str, ...]


class InMemoryKnowledgeBase:
    def __init__(self, documents: list[KnowledgeDocument] | None = None) -> None:
        self._documents = documents or default_documents()

    def search(self, query: str, limit: int = 4) -> list[EvidenceChunk]:
        query_tokens = set(_tokenize(query))
        scored: list[tuple[float, KnowledgeDocument]] = []

        for document in self._documents:
            content_tokens = set(_tokenize(document.content))
            title_tokens = set(_tokenize(document.title))
            tag_tokens = {tag.lower() for tag in document.tags}
            overlap = len(query_tokens & content_tokens)
            overlap += len(query_tokens & title_tokens) * 1.5
            overlap += len(query_tokens & tag_tokens) * 2

            if overlap <= 0:
                continue
            scored.append((overlap, document))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            EvidenceChunk(
                id=document.id,
                title=document.title,
                source=document.source,
                content=document.content,
                tags=list(document.tags),
                score=score,
            )
            for score, document in scored[:limit]
        ]


def _tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def default_documents() -> list[KnowledgeDocument]:
    return [
        KnowledgeDocument(
            id="dsm5-mdd",
            title="DSM-5 Major Depressive Episode",
            source=EvidenceSource.DSM5,
            content=(
                "Persistent low mood, diminished interest, sleep disturbance, fatigue, "
                "guilt, concentration problems, and functional impairment support a "
                "major depressive episode differential."
            ),
            tags=("depression", "mood", "sleep", "fatigue"),
        ),
        KnowledgeDocument(
            id="dsm5-gad",
            title="DSM-5 Generalized Anxiety Features",
            source=EvidenceSource.DSM5,
            content=(
                "Excessive worry across domains, restlessness, muscle tension, sleep "
                "disturbance, irritability, and difficulty controlling anxiety suggest "
                "generalized anxiety disorder."
            ),
            tags=("anxiety", "worry", "restlessness", "sleep"),
        ),
        KnowledgeDocument(
            id="dsm5-adjustment",
            title="DSM-5 Adjustment Disorder Pattern",
            source=EvidenceSource.DSM5,
            content=(
                "Symptoms emerging after an identifiable stressor with distress out of "
                "proportion to the event and impaired functioning may fit adjustment disorder."
            ),
            tags=("stressor", "adjustment", "distress", "function"),
        ),
        KnowledgeDocument(
            id="cbt-cognitive-restructuring",
            title="CBT Cognitive Restructuring",
            source=EvidenceSource.CBT,
            content=(
                "Identify automatic thoughts, examine evidence, challenge cognitive "
                "distortions, and replace them with balanced alternatives."
            ),
            tags=("cbt", "thoughts", "reframing", "distortions"),
        ),
        KnowledgeDocument(
            id="cbt-behavioral-activation",
            title="CBT Behavioral Activation",
            source=EvidenceSource.CBT,
            content=(
                "Low motivation and withdrawal can be addressed through activity scheduling, "
                "small achievable goals, and reinforcement of adaptive routines."
            ),
            tags=("cbt", "depression", "withdrawal", "motivation"),
        ),
        KnowledgeDocument(
            id="safety-crisis",
            title="Safety Escalation Guidance",
            source=EvidenceSource.SAFETY,
            content=(
                "Active suicidal intent, a specific plan, command hallucinations, or "
                "imminent danger require immediate crisis escalation and human review."
            ),
            tags=("suicide", "self-harm", "crisis", "escalation"),
        ),
    ]
