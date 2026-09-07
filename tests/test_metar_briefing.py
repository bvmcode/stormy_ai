"""Tests for METAR plot embedding in briefing markdown."""

from __future__ import annotations

import unittest

from stormy_ai.briefing import ensure_metar_image_markdown


class MetarBriefingMarkdownTests(unittest.TestCase):
    def test_ensure_metar_inserts_after_radar(self) -> None:
        text = (
            "## Current Weather\n\n"
            '<img src="/tmp/radar.png" alt="NEXRAD reflectivity" width="720" />\n'
            "Sunny and dry.\n"
        )
        result = ensure_metar_image_markdown(text, "/tmp/metar.png")
        self.assertIn('alt="METAR station models"', result)
        self.assertIn('src="/tmp/metar.png"', result)
        radar_at = result.index("NEXRAD reflectivity")
        metar_at = result.index("METAR station models")
        self.assertLess(radar_at, metar_at)

    def test_ensure_metar_inserts_under_current_weather_without_radar(self) -> None:
        text = "## Current Weather\n\nSunny and dry.\n"
        result = ensure_metar_image_markdown(text, "/tmp/metar.png")
        self.assertIn('alt="METAR station models"', result)
        self.assertIn('src="/tmp/metar.png"', result)
        self.assertLess(
            result.index("## Current Weather"),
            result.index("METAR station models"),
        )
        self.assertLess(
            result.index("METAR station models"),
            result.index("Sunny and dry."),
        )

    def test_ensure_metar_is_idempotent(self) -> None:
        text = (
            "## Current Weather\n\n"
            '<img src="/tmp/metar.png" alt="METAR station models" width="720" />\n'
        )
        result = ensure_metar_image_markdown(text, "/tmp/metar.png")
        self.assertEqual(result.count("METAR station models"), 1)


if __name__ == "__main__":
    unittest.main()
