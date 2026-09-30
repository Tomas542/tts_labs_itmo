"""Restore pause-bearing punctuation on the normalized RUSLAN metadata.

Lab 1 collapses expressive marks to a single character. Pause prediction needs the
raw clusters, rewritten into the only multi-character forms that stay in the text:
``!?``, ``?!``, and ``.!?``. Every other cluster becomes one mark. A raw en dash
becomes an em dash. Word forms are left as normalized, so the MFA join still holds.

Reads ``data/ruslan_dataset/processed_ruslan.csv`` and writes
``data/metadata_RUSLAN_22200_normalized.csv`` (no header, ``id|raw|nrm``).
"""

from __future__ import annotations

import csv
import re
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = ROOT / "data" / "ruslan_dataset" / "processed_ruslan.csv"
OUTPUT_PATH = ROOT / "data" / "metadata_RUSLAN_22200_normalized.csv"

WORD_RE = re.compile(
    r"[0-9A-Za-z\u0400-\u04FF\u0301]+(?:[-'\u2019][0-9A-Za-z\u0400-\u04FF\u0301]+)*",
)
SENTENCE_RUN_RE = re.compile(r"[.!?…]+")
DASH_RE = re.compile(r"[–—]")


def canonicalize_sentence_cluster(cluster: str) -> str:
    """Map a raw ``.!?`` cluster to ``!?``, ``?!``, ``.!?``, or one mark."""
    expanded = cluster.replace("…", "...")
    has_excl = "!" in expanded
    has_quest = "?" in expanded
    without_ellipsis = re.sub(r"\.{2,}", "", expanded)
    has_period = "." in without_ellipsis
    if has_excl and has_quest and has_period:
        return ".!?"
    if has_excl and has_quest:
        return "?!" if expanded.find("?") < expanded.find("!") else "!?"
    if has_excl:
        return "!"
    if has_quest:
        return "?"
    if "…" in cluster or expanded.count(".") >= 2:
        return "…"
    if "." in expanded:
        return "."
    return cluster


def _apply_runs(text: str, raw_runs: list[str], raw_has_dash: bool) -> str:
    """Replace sentence-mark runs in ``text`` from the paired raw runs."""
    matches = list(SENTENCE_RUN_RE.finditer(text))
    if raw_runs and len(raw_runs) == len(matches):
        chars = list(text)
        for match, raw in zip(reversed(matches), reversed(raw_runs), strict=True):
            chars[match.start() : match.end()] = list(canonicalize_sentence_cluster(raw))
        text = "".join(chars)
    if raw_has_dash:
        text = text.replace("-", "—")
    return text


def _segments(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Split ``text`` into a leading span and ``(word, tail)`` pairs."""
    matches = list(WORD_RE.finditer(text))
    if not matches:
        return text, []
    leading = text[: matches[0].start()]
    items: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        tail_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        items.append((match.group(), text[match.end() : tail_end]))
    return leading, items


def _word_key(word: str) -> str:
    return word.lower().replace("ё", "е").replace("Ё", "е")


def restore_punctuation(raw: str, normalized: str) -> str:
    """Copy canonical pause marks from ``raw`` onto ``normalized`` word forms."""
    raw_lead, raw_items = _segments(raw)
    nrm_lead, nrm_items = _segments(normalized)
    if not nrm_items:
        return normalized

    raw_keys = [_word_key(word) for word, _ in raw_items]
    nrm_keys = [_word_key(word) for word, _ in nrm_items]
    matcher = SequenceMatcher(a=raw_keys, b=nrm_keys, autojunk=False)

    lead_runs = SENTENCE_RUN_RE.findall(raw_lead)
    restored = [_apply_runs(nrm_lead, lead_runs, bool(DASH_RE.search(raw_lead)))]
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        raw_span = raw_items[i1:i2]
        nrm_span = nrm_items[j1:j2]
        raw_runs: list[str] = []
        raw_has_dash = False
        for _, tail in raw_span:
            raw_runs.extend(SENTENCE_RUN_RE.findall(tail))
            raw_has_dash = raw_has_dash or bool(DASH_RE.search(tail))
        nrm_run_count = sum(len(SENTENCE_RUN_RE.findall(tail)) for _, tail in nrm_span)
        paired = bool(raw_runs) and len(raw_runs) == nrm_run_count
        if tag == "equal" and len(raw_span) == len(nrm_span):
            for (_, raw_tail), (word, nrm_tail) in zip(raw_span, nrm_span, strict=True):
                runs = SENTENCE_RUN_RE.findall(raw_tail)
                restored.append(word)
                restored.append(_apply_runs(nrm_tail, runs, bool(DASH_RE.search(raw_tail))))
            continue
        run_cursor = 0
        for word, nrm_tail in nrm_span:
            count = len(SENTENCE_RUN_RE.findall(nrm_tail))
            runs = raw_runs[run_cursor : run_cursor + count] if paired else []
            run_cursor += count
            restored.append(word)
            restored.append(_apply_runs(nrm_tail, runs, raw_has_dash and "-" in nrm_tail))
    return "".join(restored)


def main() -> None:
    """Write the three-column metadata the pause prep script reads."""
    frame = pd.read_csv(SOURCE_PATH, sep="|")
    raw_col, nrm_col = frame.columns[1], frame.columns[2]
    frame[nrm_col] = [
        restore_punctuation(raw, nrm) for raw, nrm in zip(frame[raw_col], frame[nrm_col], strict=True)
    ]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        OUTPUT_PATH,
        sep="|",
        header=False,
        index=False,
        quoting=csv.QUOTE_NONE,
    )
    print(f"Wrote {len(frame)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
