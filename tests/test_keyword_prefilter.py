import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from digest import keyword_prefilter


def test_title_match_passes_filter():
    items = [
        {"id": "a", "title": "Deep learning for neural segmentation", "summary": "unrelated body"},
        {"id": "b", "title": "Neural networks in medical imaging", "summary": ""},
        {"id": "c", "title": "Coffee brewing methods", "summary": "nothing relevant here"},
    ]
    result = keyword_prefilter(items, ["neural"], keep_top=2)
    result_ids = {r["id"] for r in result}
    assert "a" in result_ids
    assert "b" in result_ids


def test_summary_match_passes_filter():
    items = [
        {"id": "a", "title": "An unrelated title", "summary": "We propose a novel neural architecture for imaging."},
        {"id": "b", "title": "Coffee review", "summary": "This paper discusses neural mechanisms in detail."},
        {"id": "c", "title": "Gardening tips", "summary": "Nothing of interest."},
    ]
    result = keyword_prefilter(items, ["neural"], keep_top=2)
    result_ids = {r["id"] for r in result}
    assert "a" in result_ids
    assert "b" in result_ids


def test_non_matching_paper_is_rejected():
    items = [
        {"id": "a", "title": "Neural networks overview", "summary": ""},
        {"id": "b", "title": "Deep neural models", "summary": ""},
        {"id": "c", "title": "Coffee brewing", "summary": "latte art techniques"},
    ]
    result = keyword_prefilter(items, ["neural"], keep_top=2)
    result_ids = {r["id"] for r in result}
    assert "c" not in result_ids
    assert len(result) == 2


def test_matching_is_case_insensitive():
    items = [
        {"id": "a", "title": "NEURAL networks", "summary": ""},
        {"id": "b", "title": "neural imaging", "summary": ""},
        {"id": "c", "title": "no match here", "summary": "nothing relevant"},
    ]
    result = keyword_prefilter(items, ["NeUrAl"], keep_top=2)
    result_ids = {r["id"] for r in result}
    assert "a" in result_ids
    assert "b" in result_ids


def test_empty_input_returns_empty_list():
    assert keyword_prefilter([], ["neural"], keep_top=10) == []
