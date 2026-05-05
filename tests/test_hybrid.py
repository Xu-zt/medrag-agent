"""Unit tests for RRF fusion logic (no external dependencies needed)."""
from medrag.retrieval.hybrid import _reciprocal_rank_fusion


def test_rrf_top_item_wins():
    """A doc appearing first in both lists should rank highest."""
    list1 = ["A", "B", "C"]
    list2 = ["A", "C", "D"]
    fused = dict(_reciprocal_rank_fusion([list1, list2]))
    assert fused["A"] > fused["B"]
    assert fused["A"] > fused["D"]


def test_rrf_union_of_lists():
    """All unique chunk_ids from both lists appear in the result."""
    list1 = ["A", "B"]
    list2 = ["C", "D"]
    fused = dict(_reciprocal_rank_fusion([list1, list2]))
    assert set(fused) == {"A", "B", "C", "D"}


def test_rrf_single_list_preserves_order():
    """With one list, RRF order matches the original ranking."""
    ranking = ["X", "Y", "Z"]
    fused = dict(_reciprocal_rank_fusion([ranking]))
    scores = [fused[k] for k in ["X", "Y", "Z"]]
    assert scores[0] > scores[1] > scores[2]


def test_rrf_cross_list_bonus():
    """A doc appearing in both lists beats one appearing in only one list."""
    list1 = ["A", "B"]
    list2 = ["B", "C"]  # B appears in both
    fused = dict(_reciprocal_rank_fusion([list1, list2]))
    # B appears in both lists; A and C each in one only
    assert fused["B"] > fused["A"]
    assert fused["B"] > fused["C"]


def test_rrf_empty_lists():
    """Edge case: empty lists should return empty result."""
    assert _reciprocal_rank_fusion([]) == []
    assert _reciprocal_rank_fusion([[]]) == []
