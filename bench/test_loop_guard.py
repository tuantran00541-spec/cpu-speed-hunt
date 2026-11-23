#!/usr/bin/env python3
"""Unit tests for bench/loop_guard.py. Stdlib only.

Real cases: the Q4 fp16_07 loop (100 tokens, from
results-fp16/artifacts-q4km/results-llama-fp16-q4km.json) must be caught and
cut back to the good sentence; all 21 clean leg D outputs (from
r15d-fp16-21-result/results-llama-r15d-fp16-21.json) must pass untouched.
Synthetic cases pin the "what may repeat" thresholds: Vietnamese
reduplication ("di di", "rat rat") and repeated names must survive, while a
3-word phrase 3x in a row and a sentence 2x in a row must be cut.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from loop_guard import truncate_loop

ROOT = Path(__file__).resolve().parent.parent
Q4_JSON = ROOT / "results-fp16" / "artifacts-q4km" / "results-llama-fp16-q4km.json"
R15D_JSON = ROOT / "r15d-fp16-21-result" / "results-llama-r15d-fp16-21.json"


def _images(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["images"]


class TestRealCases(unittest.TestCase):
    def test_q4_fp16_07_loop_is_caught(self):
        loop = next(im for im in _images(Q4_JSON)
                    if im["name"] == "fp16_07.png")
        self.assertEqual(loop["tokens"], 100)
        cleaned, hit, detail = truncate_loop(loop["output"])
        self.assertTrue(hit, "Q4 fp16_07 loop was not detected")
        self.assertIn("ngram", detail)
        # Good sentence intact, runaway tail gone: the loop phrase survives
        # at most once (the first, good copy), and the dangling fragment
        # left by the cut ("như là các") is trimmed back to the last
        # content word.
        self.assertTrue(cleaned.startswith("Ở trong, có lẽ có những thú vật"),
                        f"good prefix damaged: {cleaned[:80]!r}")
        self.assertTrue(cleaned.endswith("như vậy"),
                        f"kept text should end at the good sentence, got: {cleaned[-80:]!r}")
        self.assertNotIn("như là các", cleaned)
        self.assertLess(len(cleaned.split()), 40)

    def test_leg_d_21_clean_outputs_untouched(self):
        for im in _images(R15D_JSON):
            cleaned, hit, detail = truncate_loop(im["output"])
            self.assertFalse(hit, f"{im['name']} falsely flagged: {detail}")
            self.assertEqual(cleaned, im["output"])


class TestLegitRepetitionSurvives(unittest.TestCase):
    def test_reduplication(self):
        for text in ("Đi đi, nhanh nhanh lên.",
                     "Hắn nói rất rất hay.",
                     "Ta là thuộc hạ của GON MA đại nhân, đệ tử thứ tư của thiên ma.",
                     "Mu ki?",
                     "Ngươi sẽ gặp ta lần nữa. Ta sẽ đợi ngươi."):
            _, hit, detail = truncate_loop(text)
            self.assertFalse(hit, f"{text!r} falsely flagged: {detail}")


class TestLoopsAreCut(unittest.TestCase):
    def test_phrase_loop_3x(self):
        text = "trong rừng có hổ trong rừng có hổ trong rừng có hổ"
        cleaned, hit, _ = truncate_loop(text)
        self.assertTrue(hit)
        self.assertEqual(cleaned, "trong rừng có hổ")

    def test_sentence_loop_2x(self):
        text = "Tôi đi học. Tôi đi học. Tôi đi học."
        cleaned, hit, _ = truncate_loop(text)
        self.assertTrue(hit)
        self.assertEqual(cleaned, "Tôi đi học.")

    def test_two_word_loop_4x(self):
        cleaned, hit, _ = truncate_loop("ta đi ta đi ta đi ta đi")
        self.assertTrue(hit)
        self.assertEqual(cleaned, "ta đi")

    def test_word_loop_6x_never_empty(self):
        cleaned, hit, _ = truncate_loop("ha ha ha ha ha ha")
        self.assertTrue(hit)
        self.assertTrue(cleaned)


if __name__ == "__main__":
    unittest.main(verbosity=2)
