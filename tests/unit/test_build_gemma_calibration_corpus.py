import json

import build_gemma_calibration_corpus as corpus
import pytest


def test_lexical_overlap_ignores_case_punctuation_and_unicode_width():
    a = "ONE two three four five six seven eight nine ten eleven twelve thirteen fourteen."
    b = "one, two three four five six seven eight nine ten eleven twelve thirteen different"
    assert corpus.shingles(a) & corpus.shingles(b)
    assert corpus.normalized("  Ａ B\nC ") == "a b c"
    assert not corpus.shingles("unrelated short text")


def test_whole_record_selection_is_deterministic_balanced_and_excludes_overlap():
    excluded = "one two three four five six seven eight nine ten eleven twelve thirteen"
    records = [
        {"id": f"{category}-{i}", "category": category, "tokens": 100,
         "text": excluded if i == 0 else f"Unique {category} {i}"}
        for category in corpus.WEIGHTS for i in range(12)
    ]
    first = corpus.choose(records, 1000, 42, corpus.shingles(excluded))
    second = corpus.choose(list(reversed(records)), 1000, 42, corpus.shingles(excluded))
    assert first == second
    assert len({r["id"] for r in first}) == len(first)
    assert not any(r["text"] == excluded for r in first)
    for category, weight in corpus.WEIGHTS.items():
        total = sum(r["tokens"] for r in first if r["category"] == category)
        assert 1000 * weight <= total < 1000 * weight + 100
    with pytest.raises(ValueError, match="insufficient"):
        corpus.choose([], 1000, 42, set())


def test_wikitext_article_groups_and_source_text_preserved(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "train.parquet"
    pq.write_table(pa.table({"text": [" = First article = \n", "first paragraph", "second paragraph",
                                     " = Second article = \n", "third paragraph"]}), path)
    records = list(corpus.source_records("prose", path))
    assert records[1]["group"] == records[2]["group"]
    assert records[1]["group"] != records[4]["group"]
    assert records[1]["text"] == "first paragraph"
    pq.write_table(pa.table({"content": [r"Fix this code: ```\nx = 1\n```" + "\n算术与数学语言示例"]}), path)
    records = list(corpus.source_records("mixed", path))
    assert records[0]["text"] == r"Fix this code: ```\nx = 1\n```"
    assert records[0]["category"] == "code"
    assert records[1]["category"] == "multiscript"


def test_evaluation_index_reads_only_heldout_splits_and_hashes_inputs(tmp_path):
    pa = pytest.importorskip("pyarrow")
    root = tmp_path / "Salesforce___wikitext"
    root.mkdir()
    texts = {"train": "training content stays out of the exclusion index altogether",
             "test": "one two three four five six seven eight nine ten eleven twelve thirteen"}
    for split, text in texts.items():
        table = pa.table({"text": [text]})
        with (pa.OSFile(str(root / f"wikitext-{split}.arrow"), "wb") as out,
              pa.ipc.new_stream(out, table.schema) as writer):
            writer.write_table(table)
    exact, overlap, receipts = corpus.evaluation_index(tmp_path)
    assert len(receipts) == 1
    assert receipts[0]["path"].endswith("test.arrow")
    assert corpus.digest(corpus.normalized(texts["test"])) in exact
    assert corpus.digest(corpus.normalized(texts["train"])) not in exact
    assert corpus.shingles(texts["test"]) <= overlap
    assert len(receipts[0]["sha256"]) == 64
    json.dumps(receipts)
