"""Selecting and cleaning the earnings release behind an Item 2.02 8-K."""

from __future__ import annotations

import gzip

import pandas as pd
import pytest

from earnings_engine.data.filing_text import (
    FilingTextDownloader,
    choose_document,
    html_to_text,
    is_earnings_release,
    parse_index,
    read_text,
)

INDEX_PAGE = """
<table class="tableFile" summary="Document Format Files">
  <tr><th>Seq</th><th>Description</th><th>Document</th><th>Type</th><th>Size</th></tr>
  <tr><td>1</td><td>8-K</td>
      <td><a href="/Archives/edgar/data/1/2/cover.htm">cover.htm</a></td>
      <td>8-K</td><td>36645</td></tr>
  <tr><td>2</td><td>EXHIBIT 99.1</td>
      <td><a href="/Archives/edgar/data/1/2/release.htm">release.htm</a></td>
      <td>EX-99.1</td><td>218498</td></tr>
  <tr><td>3</td><td>EXHIBIT 99.2</td>
      <td><a href="/Archives/edgar/data/1/2/slides.htm">slides.htm</a></td>
      <td>EX-99.2</td><td>948012</td></tr>
  <tr><td>4</td><td></td>
      <td><a href="/Archives/edgar/data/1/2/logo.jpg">logo.jpg</a></td>
      <td>GRAPHIC</td><td>1264</td></tr>
</table>
"""


def test_index_rows_carry_type_href_and_size():
    rows = parse_index(INDEX_PAGE)
    assert {"type", "href", "size"} <= set(rows[0])
    assert [r["type"] for r in rows] == ["8-K", "EX-99.1", "EX-99.2", "GRAPHIC"]


def test_the_release_is_exhibit_99_1_not_the_biggest_exhibit():
    """99.2 here is a larger slide deck; the release is still 99.1."""
    assert choose_document(INDEX_PAGE, "cover.htm") == "release.htm"


def test_a_text_only_8k_falls_back_to_its_primary_document():
    page = INDEX_PAGE.replace("EX-99.1", "GRAPHIC").replace("EX-99.2", "GRAPHIC")
    assert choose_document(page, "cover.htm") == "cover.htm"


def test_only_item_2_02_filings_are_earnings_releases():
    assert is_earnings_release("2.02,9.01")
    assert is_earnings_release("1.01, 2.02")
    assert not is_earnings_release("5.02,9.01")
    assert not is_earnings_release(None)
    # 2.02 must not be matched by a prefix of a different item.
    assert not is_earnings_release("2.03")


def test_markup_scripts_and_entities_are_removed():
    raw = (
        "<html><head><style>p{color:red}</style><script>x=1</script></head>"
        "<body><p>Revenue rose&nbsp;12%.</p><p>Margins&amp;mix improved.</p></body></html>"
    )
    text = html_to_text(raw)
    assert "color:red" not in text and "x=1" not in text
    assert "Revenue rose 12%." in text
    assert "Margins&mix improved." in text
    assert "<" not in text


def test_block_elements_become_line_breaks_not_joined_words():
    assert "quarterresults" not in html_to_text("<p>quarter</p><p>results</p>").lower()


class FakeClient:
    def __init__(self, pages):
        self.pages = pages
        self.requested = []

    def get_text(self, url, **_kwargs):
        self.requested.append(url)
        for fragment, body in self.pages.items():
            if fragment in url:
                return body
        raise AssertionError(f"unexpected url {url}")


def a_downloader(tmp_path, body="<p>%s</p>" % ("Results were strong. " * 60)):
    client = FakeClient({"-index.htm": INDEX_PAGE, "release.htm": body})
    return FilingTextDownloader(client=client, text_dir=tmp_path), client


def test_fetch_writes_a_gzipped_cache_and_reuses_it(tmp_path):
    downloader, client = a_downloader(tmp_path)
    first = downloader.fetch(1, "0000000000-20-000001", "cover.htm")
    assert "Results were strong." in first
    assert downloader.path_for("0000000000-20-000001").exists()
    calls = len(client.requested)
    assert downloader.fetch(1, "0000000000-20-000001", "cover.htm") == first
    assert len(client.requested) == calls, "a cached release must not be re-requested"


def test_a_stub_document_is_rejected_rather_than_cached(tmp_path):
    downloader, _client = a_downloader(tmp_path, body="<p>See exhibit.</p>")
    with pytest.raises(ValueError, match="too short"):
        downloader.fetch(1, "0000000000-20-000002", "cover.htm")
    assert not downloader.path_for("0000000000-20-000002").exists()


def test_fetch_many_skips_filings_that_are_not_earnings_releases(tmp_path):
    downloader, _client = a_downloader(tmp_path)
    filings = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "AAA"],
            "cik": [1, 1, 1],
            "accession": ["0-20-1", "0-20-2", "0-20-3"],
            "form": ["8-K", "8-K", "10-Q"],
            "items": ["2.02,9.01", "5.02", ""],
            "filing_date": pd.to_datetime(["2020-01-01", "2020-04-01", "2020-07-01"]),
            "primary_document": ["cover.htm"] * 3,
        }
    )
    counts = downloader.fetch_many(filings)
    assert counts == {"requested": 1, "downloaded": 1, "cached": 0, "failed": 0}


def test_a_single_broken_filing_does_not_abort_the_job(tmp_path):
    downloader, _client = a_downloader(tmp_path)
    filings = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "cik": [1, 1],
            "accession": ["0-20-1", "0-20-2"],
            "form": ["8-K", "8-K"],
            "items": ["2.02", "2.02"],
            "filing_date": pd.to_datetime(["2020-01-01", "2020-04-01"]),
            "primary_document": ["cover.htm", "cover.htm"],
        }
    )
    downloader.client.pages["release.htm"] = "<p>short</p>"
    counts = downloader.fetch_many(filings)
    assert counts["failed"] == 2 and counts["downloaded"] == 0


def test_read_text_handles_both_the_gzipped_and_plain_layouts(tmp_path):
    with gzip.open(tmp_path / "a.txt.gz", "wt", encoding="utf-8") as handle:
        handle.write("gzipped")
    (tmp_path / "b.txt").write_text("plain", encoding="utf-8")
    assert read_text(tmp_path, "a") == "gzipped"
    assert read_text(tmp_path, "b") == "plain"
    assert read_text(tmp_path, "missing") == ""
