from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

from psyche_agent_dx.schemas import EvidenceChunk, EvidenceSource


TOKEN_PATTERN = re.compile(r"[A-Za-z]{2,}|[\u4e00-\u9fff]{2,}")
BM25_K1 = 1.5
BM25_B = 0.75
TITLE_WEIGHT = 1.3
TAG_WEIGHT = 1.7
CONTENT_WEIGHT = 1.0


@dataclass(frozen=True)
class KnowledgeDocument:
    id: str
    title: str
    source: EvidenceSource
    content: str
    tags: tuple[str, ...]


@dataclass(frozen=True)
class _IndexedDocument:
    document: KnowledgeDocument
    content_tokens: list[str]
    title_tokens: list[str]
    tag_tokens: list[str]
    content_term_freqs: dict[str, int]
    title_term_freqs: dict[str, int]
    tag_term_freqs: dict[str, int]


class InMemoryKnowledgeBase:
    """Local knowledge base backed by a persistent JSONL corpus and BM25 scoring."""

    def __init__(
        self,
        documents: list[KnowledgeDocument] | None = None,
        *,
        corpus_path: str | Path | None = None,
    ) -> None:
        if documents is not None:
            raw_documents = documents
        else:
            raw_documents = load_documents(corpus_path or default_corpus_path())

        self._documents = raw_documents
        self._indexed_documents = [_index_document(document) for document in raw_documents]
        self._content_doc_freqs = _document_frequencies(
            item.content_term_freqs.keys() for item in self._indexed_documents
        )
        self._title_doc_freqs = _document_frequencies(
            item.title_term_freqs.keys() for item in self._indexed_documents
        )
        self._tag_doc_freqs = _document_frequencies(
            item.tag_term_freqs.keys() for item in self._indexed_documents
        )
        self._avg_content_length = _average_length(
            len(item.content_tokens) for item in self._indexed_documents
        )
        self._avg_title_length = _average_length(len(item.title_tokens) for item in self._indexed_documents)
        self._avg_tag_length = _average_length(len(item.tag_tokens) for item in self._indexed_documents)

    def search(self, query: str, limit: int = 4) -> list[EvidenceChunk]:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scored: list[tuple[float, KnowledgeDocument]] = []
        total_docs = len(self._indexed_documents)

        for indexed in self._indexed_documents:
            score = 0.0
            score += CONTENT_WEIGHT * _bm25_score(
                query_tokens,
                indexed.content_term_freqs,
                self._content_doc_freqs,
                len(indexed.content_tokens),
                self._avg_content_length,
                total_docs,
            )
            score += TITLE_WEIGHT * _bm25_score(
                query_tokens,
                indexed.title_term_freqs,
                self._title_doc_freqs,
                len(indexed.title_tokens),
                self._avg_title_length,
                total_docs,
            )
            score += TAG_WEIGHT * _bm25_score(
                query_tokens,
                indexed.tag_term_freqs,
                self._tag_doc_freqs,
                len(indexed.tag_tokens),
                self._avg_tag_length,
                total_docs,
            )
            score += _tag_overlap_bonus(query_tokens, indexed.document.tags)

            if score <= 0:
                continue
            scored.append((score, indexed.document))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            EvidenceChunk(
                id=document.id,
                title=document.title,
                source=document.source,
                content=document.content,
                tags=list(document.tags),
                score=round(score, 4),
            )
            for score, document in scored[:limit]
        ]


def default_corpus_path() -> Path:
    return project_root() / "data" / "knowledge" / "default_corpus.jsonl"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_documents(corpus_path: str | Path) -> list[KnowledgeDocument]:
    path = Path(corpus_path)
    documents: list[KnowledgeDocument] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            try:
                documents.append(
                    KnowledgeDocument(
                        id=str(payload["id"]),
                        title=str(payload["title"]),
                        source=EvidenceSource(str(payload["source"])),
                        content=str(payload["content"]),
                        tags=tuple(str(tag) for tag in payload.get("tags", [])),
                    )
                )
            except KeyError as exc:
                raise ValueError(f"Invalid corpus record at {path}:{line_number}: missing {exc.args[0]}") from exc

    return documents


def _index_document(document: KnowledgeDocument) -> _IndexedDocument:
    content_tokens = _tokenize(document.content)
    title_tokens = _tokenize(document.title)
    tag_tokens = [token for tag in document.tags for token in _tokenize(tag)]
    return _IndexedDocument(
        document=document,
        content_tokens=content_tokens,
        title_tokens=title_tokens,
        tag_tokens=tag_tokens,
        content_term_freqs=_term_frequencies(content_tokens),
        title_term_freqs=_term_frequencies(title_tokens),
        tag_term_freqs=_term_frequencies(tag_tokens),
    )


def _document_frequencies(term_sets: Iterable[Iterable[str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for terms in term_sets:
        for term in set(terms):
            counts[term] = counts.get(term, 0) + 1
    return counts


def _average_length(lengths: Iterable[int]) -> float:
    values = list(lengths)
    if not values:
        return 1.0
    return max(sum(values) / len(values), 1.0)


def _term_frequencies(tokens: Iterable[str]) -> dict[str, int]:
    freqs: dict[str, int] = {}
    for token in tokens:
        freqs[token] = freqs.get(token, 0) + 1
    return freqs


def _bm25_score(
    query_tokens: Iterable[str],
    term_freqs: dict[str, int],
    doc_freqs: dict[str, int],
    doc_length: int,
    avg_doc_length: float,
    total_docs: int,
) -> float:
    score = 0.0
    normalized_length = 1 - BM25_B + BM25_B * (doc_length / max(avg_doc_length, 1.0))

    for token in query_tokens:
        frequency = term_freqs.get(token, 0)
        if frequency <= 0:
            continue
        doc_frequency = doc_freqs.get(token, 0)
        inverse_doc_frequency = math.log(1 + ((total_docs - doc_frequency + 0.5) / (doc_frequency + 0.5)))
        numerator = frequency * (BM25_K1 + 1)
        denominator = frequency + BM25_K1 * normalized_length
        score += inverse_doc_frequency * (numerator / denominator)

    return score


def _tag_overlap_bonus(query_tokens: Iterable[str], tags: tuple[str, ...]) -> float:
    lowered_tags = {tag.lower() for tag in tags}
    return sum(0.25 for token in query_tokens if token in lowered_tags)


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for match in TOKEN_PATTERN.finditer(text):
        token = match.group(0).lower()
        tokens.append(token)
        if _is_cjk_token(token):
            tokens.extend(_cjk_ngrams(token, 2))
            tokens.extend(_cjk_ngrams(token, 3))
    return tokens


def _is_cjk_token(token: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in token)


def _cjk_ngrams(token: str, size: int) -> list[str]:
    if len(token) <= size:
        return []
    return [token[index : index + size] for index in range(len(token) - size + 1)]
