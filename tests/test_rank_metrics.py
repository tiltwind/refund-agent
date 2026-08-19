"""排序指标的判分口径回归测试。

判分口径的改动会让历史 run 的分数不再可比，所以这里钉死三件事：跨块样本按最深的
那个算、掉出证据列表记 0（而不是跳过）、没有 source 的样本不产生指标。
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rag" / "experiments" / "rag-ex-1"))

import rank_metrics  # noqa: E402

CANDIDATES = ["a", "b", "c", "d", "e"]
EVIDENCE = ["b", "a", "c"]


class RankMetricsTests(unittest.TestCase):
    def test_single_source_takes_its_own_rank(self):
        got = rank_metrics.rank_metrics(["b"], CANDIDATES, EVIDENCE)
        self.assertEqual(got, {"candidate_hit": 1.0, "hit@1": 1.0, "hit@4": 1.0, "mrr": 1.0})

    def test_cross_chunk_source_takes_the_deepest_rank(self):
        # 两段条文分别排第 2 和第 3，按第 3 算 —— 只召回一半答不全
        got = rank_metrics.rank_metrics(["a", "c"], CANDIDATES, EVIDENCE)
        self.assertEqual(got["mrr"], round(1 / 3, 3))
        self.assertEqual(got["hit@1"], 0.0)

    def test_dropped_by_threshold_scores_zero_but_keeps_candidate_hit(self):
        got = rank_metrics.rank_metrics(["d"], CANDIDATES, EVIDENCE)
        self.assertEqual(got["candidate_hit"], 1.0)
        self.assertEqual(got["hit@4"], 0.0)
        self.assertIn("卡在重排", rank_metrics.explain(["d"], CANDIDATES, EVIDENCE))

    def test_missed_by_recall_is_distinguishable(self):
        got = rank_metrics.rank_metrics(["z"], CANDIDATES, EVIDENCE)
        self.assertEqual(got["candidate_hit"], 0.0)
        self.assertIn("卡在召回层", rank_metrics.explain(["z"], CANDIDATES, EVIDENCE))

    def test_no_source_yields_no_metrics(self):
        # 空字典而不是一堆 0：这条样本没有这几个指标，不是判负
        self.assertEqual(rank_metrics.rank_metrics([], CANDIDATES, EVIDENCE), {})


if __name__ == "__main__":
    unittest.main()
