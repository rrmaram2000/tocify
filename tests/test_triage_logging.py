import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import digest


EXPECTED_FIELDS = {
    "ts",
    "batch_index",
    "total_batches",
    "batch_size",
    "approx_input_tokens",
    "wall_clock_seconds",
    "score_distribution",
}

EXPECTED_DIST_FIELDS = {"count", "min", "max", "mean", "buckets"}


def _fake_call_openai_triage(_client, _interests, items):
    return {
        "week_of": "2026-04-14",
        "notes": "",
        "ranked": [
            {
                "id": it["id"],
                "title": it["title"],
                "link": it.get("link", ""),
                "source": it.get("source", ""),
                "published_utc": it.get("published_utc"),
                "score": 0.5,
                "why": "mock",
                "tags": [],
            }
            for it in items
        ],
    }


def test_triage_batches_writes_jsonl_log(monkeypatch, tmp_path):
    monkeypatch.setattr(digest, "TRIAGE_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(digest, "call_openai_triage", _fake_call_openai_triage)

    items = [
        {
            "id": f"id{i}",
            "title": f"Title {i}",
            "link": "http://example.com",
            "source": "test-feed",
            "published_utc": None,
            "summary": "a short summary",
        }
        for i in range(5)
    ]

    result = digest.triage_in_batches(
        client=None,
        interests={"keywords": [], "narrative": ""},
        items=items,
        batch_size=2,
    )
    assert len(result["ranked"]) == 5

    log_files = list(tmp_path.glob("triage-*.jsonl"))
    assert len(log_files) == 1, f"expected exactly one log file, got {log_files}"

    lines = log_files[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3, "expected 3 batch entries for 5 items @ batch_size=2"

    for raw in lines:
        entry = json.loads(raw)
        assert EXPECTED_FIELDS.issubset(entry.keys()), f"missing fields: {EXPECTED_FIELDS - entry.keys()}"
        assert isinstance(entry["batch_size"], int) and entry["batch_size"] > 0
        assert isinstance(entry["approx_input_tokens"], int) and entry["approx_input_tokens"] >= 0
        assert isinstance(entry["wall_clock_seconds"], (int, float)) and entry["wall_clock_seconds"] >= 0
        dist = entry["score_distribution"]
        assert EXPECTED_DIST_FIELDS.issubset(dist.keys())
        assert dist["count"] == entry["batch_size"]
