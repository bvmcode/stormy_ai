"""Tests for stripping duplicate LLM briefing titles and preamble."""

from __future__ import annotations

import unittest

from stormy_ai.briefing import strip_llm_briefing_preamble


class BriefingPreambleTests(unittest.TestCase):
    def test_strips_duplicate_title_and_issued_for(self) -> None:
        text = (
            "# Weather Briefing: Atco, NJ 08004\n"
            "**Issued for:** 39.77°N, 74.89°W — briefing cycle 2026-08-29\n"
            "\n"
            "---\n"
            "\n"
            "## Headline\n"
            "Dry and pleasant.\n"
        )
        stripped = strip_llm_briefing_preamble(text)
        self.assertTrue(stripped.startswith("## Headline"))
        self.assertNotIn("# Weather Briefing", stripped)

    def test_strips_status_preamble_and_em_dash_title(self) -> None:
        text = (
            "All tools have returned successfully. "
            "Here is the complete weather briefing for Atco, NJ.\n"
            "\n"
            "---\n"
            "\n"
            "# Weather Briefing — Atco, NJ 08004 (Camden County)\n"
            "\n"
            "## Headline\n"
            "Storms possible.\n"
        )
        stripped = strip_llm_briefing_preamble(text)
        self.assertEqual(
            stripped,
            "## Headline\nStorms possible.\n",
        )

    def test_strips_for_title_variant(self) -> None:
        text = "# Weather Briefing for Atco, NJ 08004\n\n## Headline\n\nRain.\n"
        stripped = strip_llm_briefing_preamble(text)
        self.assertEqual(stripped, "## Headline\n\nRain.\n")

    def test_leaves_clean_body_unchanged(self) -> None:
        text = "## Headline\n\nClear skies.\n\n## Active Alerts\n\nNone.\n"
        self.assertEqual(strip_llm_briefing_preamble(text), text)


if __name__ == "__main__":
    unittest.main()
