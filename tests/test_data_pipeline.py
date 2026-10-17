"""Data pipeline tests: cleaning and dataset-version behaviour."""

import pandas as pd
import pytest

from ml.data_pipeline import build_v1, build_v2, clean_text


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("<br />Hello", "Hello"),
        ("<b>bold</b> text", "bold text"),
        ("Visit https://example.com now", "Visit now"),
        ("Check www.example.com", "Check"),
        ("  spaced    out  ", "spaced out"),
        ("", ""),
        ("     ", ""),
        (None, ""),
        (123, ""),
    ],
)
def test_clean_text(raw, expected):
    assert clean_text(raw) == expected


def test_build_v1_drops_null_and_blank_rows():
    raw = pd.DataFrame(
        {"text": ["good one", None, "   ", "another"], "label": [1, 0, 1, 0]}
    )
    result = build_v1(raw)
    assert list(result["text"]) == ["good one", "another"]
    assert list(result["label"]) == [1, 0]


def test_build_v2_removes_duplicates_and_extreme_lengths():
    just_right = "word " * 20  # 20 words -> kept
    too_short = "word " * 5  # below min_words
    too_long = "word " * 500  # above max_words

    raw = pd.DataFrame(
        {
            "text": [just_right, just_right, too_short, too_long],
            "label": [1, 1, 0, 0],
        }
    )
    result = build_v2(raw)
    assert len(result) == 1
    assert result["text"].iloc[0].startswith("word word")
