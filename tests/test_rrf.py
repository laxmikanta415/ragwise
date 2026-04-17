"""Tests for rrf() — Reciprocal Rank Fusion — S1-T4."""
from __future__ import annotations

from ragwise.utils import rrf


def test_single_list_preserves_order() -> None:
    result = rrf([["a", "b", "c"]])
    ids = [doc_id for doc_id, _ in result]
    assert ids == ["a", "b", "c"]
    scores = [score for _, score in result]
    assert scores[0] > scores[1] > scores[2]


def test_two_identical_lists_doubles_scores() -> None:
    single = rrf([["a", "b", "c"]])
    double = rrf([["a", "b", "c"], ["a", "b", "c"]])
    single_scores = {doc_id: score for doc_id, score in single}
    double_scores = {doc_id: score for doc_id, score in double}
    for doc_id in ["a", "b", "c"]:
        assert abs(double_scores[doc_id] - 2 * single_scores[doc_id]) < 1e-10
    # Order is preserved
    assert [d for d, _ in double] == ["a", "b", "c"]


def test_two_disjoint_lists_all_appear() -> None:
    result = rrf([["a", "b"], ["c", "d"]])
    ids = {doc_id for doc_id, _ in result}
    assert ids == {"a", "b", "c", "d"}


def test_doc_in_both_lists_ranks_higher_than_doc_in_one() -> None:
    # "a" appears at rank 1 in both lists; "b" appears at rank 1 in only one list
    result = rrf([["a", "b"], ["a", "c"]])
    scores = {doc_id: score for doc_id, score in result}
    assert scores["a"] > scores["b"]
    assert scores["a"] > scores["c"]


def test_empty_input_returns_empty() -> None:
    assert rrf([]) == []


def test_empty_inner_lists_returns_empty() -> None:
    assert rrf([[], []]) == []


def test_k_parameter_larger_k_scores_closer_together() -> None:
    result_k10 = rrf([["a", "b", "c"]], k=10)
    result_k200 = rrf([["a", "b", "c"]], k=200)
    scores_k10 = [score for _, score in result_k10]
    scores_k200 = [score for _, score in result_k200]
    # Spread between rank 1 and rank 3 should be larger for small k
    spread_k10 = scores_k10[0] - scores_k10[2]
    spread_k200 = scores_k200[0] - scores_k200[2]
    assert spread_k10 > spread_k200


def test_tie_breaking_is_alphabetical() -> None:
    # "b" and "z" both at rank 1 in their own list — same score, tie broken by name
    result = rrf([["z"], ["b"]])
    ids = [doc_id for doc_id, _ in result]
    assert ids == ["b", "z"]


def test_score_formula_correct() -> None:
    # Single doc at rank 1 with k=60: score = 1/(60+1) = 1/61
    result = rrf([["x"]], k=60)
    assert len(result) == 1
    assert abs(result[0][1] - 1 / 61) < 1e-10
