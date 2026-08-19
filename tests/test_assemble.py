"""装配分组口径的回归测试。"""

import unittest
from types import SimpleNamespace

from rag.retrieving.pipeline import assemble


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


if __name__ == "__main__":
    unittest.main()
