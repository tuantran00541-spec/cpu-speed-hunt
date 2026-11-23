#!/usr/bin/env python3
"""Loop guard for Qwen3.5-0.8B greedy generations.

Greedy decoding can fall into an attractor: once a phrase repeats, repeating
it again becomes the highest-probability continuation, so the model loops to
the token cap. This module detects that *after* generation and truncates the
output at the loop start. It changes nothing on clean outputs.

Two rules, both word-level (Vietnamese is whitespace-separated):

1. Repeated 4-gram: the same 4-word phrase occurs twice with the two
   occurrences at most 8 words apart (a loop repeats with a short period;
   two far-apart occurrences are legitimate parallel phrasing, not a loop).
   Cut at the START of the second occurrence, so the first (good) copy is
   always kept. This encodes "what may repeat": Vietnamese reduplication
   ("di di", "rat rat") and repeated names are 1-2 words and can never form
   a repeated 4-gram; a 4-word phrase twice in a row inside one bubble is
   never legitimate. Any periodic loop (period 1-8 words) contains a
   repeated 4-gram, so this one rule catches every loop shape.
2. Sentence loop: the same sentence twice in a row -> cut before the 2nd.
   (Covers short-sentence loops too short to form a 4-gram.)

A cut can land mid-cycle: the loop's period is only known up to rotation,
so the kept text can end with a dangling fragment such as "nhu la cac"
(the first words of the next period). A truncated result is therefore
trimmed back past trailing Vietnamese function words (nhu, la, cac, cua,
va, ...) — words that can never end a well-formed sentence — so the
result ends at the last content word. This trim runs ONLY on truncated
text; clean outputs are never touched.

Truncation never yields an empty result: rule 1 keeps at least the first
4-gram, rule 2 keeps at least the first sentence.

Stdlib only. Used by bench/llama_r15d_fp16.py (records raw + cleaned +
truncated) and unit-tested by bench/test_loop_guard.py.
"""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"\S+")
_SENT_RE = re.compile(r"[^.!?\u2026]+[.!?\u2026]+")
_STRIP_CHARS = ".,!?;:\u2026\"'\"\"''()[]"
NGRAM = 4
MAX_GAP = 8
# Vietnamese function words that can never end a well-formed sentence.
# Only used to trim a dangling fragment left by a loop cut (see module doc).
DANGLING = frozenset(
    "như là các của và thì mà để với trong những một đã đang sẽ ở tại "
    "cho đến từ về bởi nên nhưng hoặc nếu khi cũng bị".split()
)


def _trim_dangling(text: str) -> str:
    spans = [(m.group(0), m.start(), m.end()) for m in _WORD_RE.finditer(text)]
    end = len(spans)
    while end > 0 and _norm(spans[end - 1][0]) in DANGLING:
        end -= 1
    if end == 0:
        return text
    return text[:spans[end - 1][2]].rstrip()


def _norm(word: str) -> str:
    return word.strip(_STRIP_CHARS).lower()


def _ngram_cut(words: list[str]) -> int | None:
    """Word index where a repeated 4-gram's second occurrence starts."""
    normed = [_norm(w) for w in words]
    last_seen: dict[tuple[str, ...], int] = {}
    for i in range(len(words) - NGRAM + 1):
        key = tuple(normed[i:i + NGRAM])
        if not any(key):
            continue
        prev = last_seen.get(key)
        if prev is not None and i - prev <= MAX_GAP:
            return i
        last_seen[key] = i
    return None


def _sentence_cut(text: str) -> int | None:
    """Char index where a repeated sentence starts, else None."""
    sentences = [(m.group(0), m.start()) for m in _SENT_RE.finditer(text)]
    for (s1, _), (s2, start2) in zip(sentences, sentences[1:]):
        if s1.strip().lower() == s2.strip().lower() and len(s1.strip()) > 1:
            return start2
    return None


def truncate_loop(text: str) -> tuple[str, bool, str]:
    """Cut a repetition loop out of generated text.

    Returns (cleaned_text, was_truncated, detail). detail names the rule
    and the cut point, e.g. "ngram: word 39" or "sentence: char 12", or ""
    when clean.
    """
    if not text or not text.strip():
        return text, False, ""
    spans = [(m.group(0), m.start(), m.end()) for m in _WORD_RE.finditer(text)]
    words = [w for w, _, _ in spans]
    cut_word = _ngram_cut(words)
    cut_char: int | None = None
    detail = ""
    if cut_word is not None:
        cut_char = spans[cut_word][1]
        detail = f"ngram: word {cut_word}"
    sent_char = _sentence_cut(text)
    if sent_char is not None and (cut_char is None or sent_char < cut_char):
        cut_char = sent_char
        detail = f"sentence: char {sent_char}"
    if cut_char is None:
        return text, False, ""
    cleaned = _trim_dangling(text[:cut_char].rstrip())
    return cleaned, True, detail


if __name__ == "__main__":
    import sys
    for path in sys.argv[1:]:
        raw = open(path, encoding="utf-8").read()
        cleaned, hit, detail = truncate_loop(raw)
        print(f"{path}: truncated={hit} {detail}")
        print(cleaned)
