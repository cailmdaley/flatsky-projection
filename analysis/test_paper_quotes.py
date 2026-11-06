"""Regression tests for the paper's source-backed quotation gate."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from check_paper_quotes import AUDIT, ROOT, check, normalize


class QuoteChecks(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads((AUDIT / "quotes.json").read_text())
        self.tex = (ROOT / "paper/main.tex").read_text()

    def test_all_registered_source_quotes(self):
        # The recommended capitalization repair must pass against real sources.
        tex = self.tex.replace("``the stereographic projection,", "``The stereographic projection,")
        quotes, errors = check(tex, self.manifest)
        self.assertEqual(errors, [])
        self.assertEqual(len(quotes), 10)

    def test_word_and_case_changes_fail(self):
        for quote in ["Comparable or sometimes lower cost", "comparable or always lower cost"]:
            _, errors = check("``" + quote + "''", self.manifest)
            self.assertTrue(errors)

    def test_registered_but_fabricated_quote_fails(self):
        manifest = copy.deepcopy(self.manifest)
        manifest[0]["quote"] = "comparable or always lower cost"
        _, errors = check("``comparable or always lower cost''", manifest)
        self.assertTrue(any("does not match" in e for e in errors))

    def test_source_tampering_fails(self):
        record = copy.deepcopy(self.manifest[0])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record["source"] = "source.txt"
            (root / record["source"]).write_text("changed source")
            _, errors = check("``" + record["quote"] + "''", [record], root)
            self.assertTrue(any("hash changed" in e for e in errors))

    def test_only_typographic_normalization(self):
        self.assertEqual(normalize("$\\ell$ ﬁeld cor-\nrelations"), normalize("ℓ field correlations"))
        self.assertNotEqual(normalize("The field"), normalize("the field"))
        self.assertNotEqual(normalize("bin-to-bin"), normalize("bin to bin"))


if __name__ == "__main__":
    unittest.main()
