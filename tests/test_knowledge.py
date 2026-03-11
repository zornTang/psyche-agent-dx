from __future__ import annotations

import json
import sys
from pathlib import Path
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from psyche_agent_dx.knowledge import InMemoryKnowledgeBase, default_corpus_path, load_documents


class KnowledgeBaseSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.knowledge_base = InMemoryKnowledgeBase()

    def test_default_corpus_file_exists_and_loads(self) -> None:
        documents = load_documents(default_corpus_path())

        self.assertGreaterEqual(len(documents), 10)

    def test_chinese_query_matches_gad_entry(self) -> None:
        results = self.knowledge_base.search("患者长期过度担心 紧张 睡眠障碍", limit=5)
        ids = [item.id for item in results]

        self.assertIn("dsm5-zh-gad", ids)

    def test_chinese_query_matches_adjustment_entry(self) -> None:
        results = self.knowledge_base.search("失业后的压力事件 导致功能受损 适应困难", limit=5)
        ids = [item.id for item in results]

        self.assertIn("dsm5-zh-adjustment", ids)

    def test_chinese_query_matches_social_anxiety_entry(self) -> None:
        results = self.knowledge_base.search("害怕被别人负面评价 回避社交 演讲紧张", limit=5)
        ids = [item.id for item in results]

        self.assertIn("dsm5-zh-social-anxiety", ids)

    def test_chinese_query_matches_psychosis_entry(self) -> None:
        results = self.knowledge_base.search("出现妄想 幻觉 言语紊乱 功能下降", limit=5)
        ids = [item.id for item in results]

        self.assertIn("dsm5-zh-schizophrenia", ids)

    def test_chinese_query_matches_illness_anxiety_entry(self) -> None:
        results = self.knowledge_base.search("总担心自己得了重病 反复检查 身体症状很轻", limit=5)
        ids = [item.id for item in results]

        self.assertIn("dsm5-zh-illness-anxiety", ids)

    def test_can_load_custom_jsonl_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus_path = Path(temp_dir) / "custom.jsonl"
            records = [
                {
                    "id": "custom-1",
                    "title": "Custom Anxiety Chunk",
                    "source": "dsm5",
                    "content": "This custom chunk describes persistent worry and sleep problems.",
                    "tags": ["custom", "worry"],
                },
                {
                    "id": "custom-2",
                    "title": "Custom Psychosis Chunk",
                    "source": "dsm5",
                    "content": "This custom chunk describes delusions and hallucinations.",
                    "tags": ["custom", "psychosis"],
                },
            ]
            with corpus_path.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record) + "\n")

            knowledge_base = InMemoryKnowledgeBase(corpus_path=corpus_path)
            results = knowledge_base.search("persistent worry sleep", limit=2)

            self.assertEqual(results[0].id, "custom-1")


if __name__ == "__main__":
    unittest.main()
