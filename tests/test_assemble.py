"""装配去重和重复率口径的回归测试。"""

import unittest
import sys
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "rag/experiments/rag-ex-1"))
import scorers
from rag.retrieving.pipeline import assemble
from rag.retrieving.protocol import PolicySection


def evidence(parent_id: str, chunk_id: str, parent_seq: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        row={
            "parent_id": parent_id,
            "chunk_id": chunk_id,
            "parent_seq": parent_seq,
            "section_path": f"section-{chunk_id}",
            "doc_id": "P01",
        }
    )


class AssembleGroupingTests(unittest.TestCase):
    def test_same_parent_is_one_group_even_when_section_path_differs(self):
        groups = assemble._group(
            [evidence("P#1", "P#1:00"), evidence("P#1", "P#1:01")], top_k=4
        )
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].hit_chunks, {"P#1:00", "P#1:01"})

    def test_different_parents_remain_distinct(self):
        groups = assemble._group(
            [evidence("P#1", "P#1:00"), evidence("P#2", "P#2:00", parent_seq=3)], top_k=4
        )
        self.assertEqual(len(groups), 2)


class DuplicateRatioTests(unittest.TestCase):
    def test_duplicate_ratio_is_content_signal(self):
        sections = [
            PolicySection(section="a", text="same paragraph\n\nunique"),
            PolicySection(section="b", text="same paragraph"),
        ]
        self.assertAlmostEqual(scorers.duplicate_ratio(sections), 14 / 34, places=3)


if __name__ == "__main__":
    unittest.main()
