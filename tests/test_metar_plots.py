"""Tests for METAR station-model helpers."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from stormy_ai.tools.metar import (
    _cloud_cover_oktas,
    _observation_record,
    _pressure_station_code,
    metar_plot_s3_uri,
    plot_extent_from_radius,
)


class MetarPlotHelpersTest(unittest.TestCase):
    def test_extent_matches_radar_framing(self) -> None:
        # Same formula as plot_nexrad_level2: radius * 1.3, then lat/lon deltas.
        west, east, south, north = plot_extent_from_radius(39.77, -74.89, 100.0)
        self.assertAlmostEqual(north - south, 2.0 * 130.0 / 111.0, places=5)
        self.assertLess(west, -74.89)
        self.assertGreater(east, -74.89)

    def test_pressure_station_code(self) -> None:
        self.assertEqual(_pressure_station_code(1014.5), "145")
        self.assertEqual(_pressure_station_code(998.2), "982")

    def test_cloud_cover_uses_highest_layer(self) -> None:
        layers = [
            {"amount": "SCT", "base": {"value": 1500}},
            {"amount": "OVC", "base": {"value": 3000}},
        ]
        self.assertEqual(_cloud_cover_oktas(layers), 8)
        self.assertEqual(_cloud_cover_oktas([{"amount": "CLR"}]), 0)
        self.assertIsNone(_cloud_cover_oktas([]))

    def test_s3_uri_uses_utc_time(self) -> None:
        when = datetime(2026, 9, 6, 21, 5, tzinfo=timezone.utc)
        self.assertEqual(
            metar_plot_s3_uri(when),
            "s3://stormy-ai-files/metar/2026-09-06/21_05.png",
        )

    def test_observation_record_converts_units(self) -> None:
        station = {
            "station_id": "KPHL",
            "name": "Philadelphia",
            "longitude": -75.25,
            "latitude": 39.87,
            "distance_km": 20.0,
        }
        observation = {
            "geometry": {"type": "Point", "coordinates": [-75.25, 39.87]},
            "properties": {
                "timestamp": "2026-09-06T21:05:00+00:00",
                "temperature": {"value": 27.0},
                "dewpoint": {"value": 13.0},
                "windSpeed": {"value": 16.668},
                "windDirection": {"value": 320.0},
                "seaLevelPressure": {"value": None},
                "barometricPressure": {"value": 101456.2},
                "cloudLayers": [{"amount": "CLR", "base": {"value": 3810}}],
                "textDescription": "Clear",
            },
        }

        record = _observation_record(
            station,
            observation,
            now=datetime(2026, 9, 6, 21, 10, tzinfo=timezone.utc),
        )

        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record["temperature_f"], 81)
        self.assertEqual(record["dewpoint_f"], 55)
        self.assertEqual(record["pressure_mb"], 1014.6)
        self.assertEqual(record["sky_cover_oktas"], 0)
        self.assertIsNotNone(record["wind_u_kt"])
        self.assertIsNotNone(record["wind_v_kt"])

    def test_stale_observation_is_rejected(self) -> None:
        station = {
            "station_id": "KPHL",
            "name": "Philadelphia",
            "longitude": -75.25,
            "latitude": 39.87,
            "distance_km": 20.0,
        }
        observation = {
            "geometry": {"type": "Point", "coordinates": [-75.25, 39.87]},
            "properties": {
                "timestamp": "2026-09-06T10:00:00+00:00",
                "temperature": {"value": 20.0},
                "dewpoint": {"value": 10.0},
            },
        }
        record = _observation_record(
            station,
            observation,
            now=datetime(2026, 9, 6, 21, 0, tzinfo=timezone.utc),
        )
        self.assertIsNone(record)

    @patch("stormy_ai.tools.metar.s3_uploads_enabled", return_value=False)
    @patch("stormy_ai.tools.metar.fetch_nearby_metar_observations")
    @patch("stormy_ai.tools.metar.render_metar_station_plot")
    def test_tool_returns_markdown_image_url(
        self,
        mock_render,
        mock_fetch,
        _mock_uploads,
    ) -> None:
        from pathlib import Path

        from stormy_ai.tools.metar import plot_metar_observations

        mock_fetch.return_value = [
            {
                "station_id": "KPHL",
                "name": "Philadelphia",
                "longitude": -75.25,
                "latitude": 39.87,
                "distance_km": 20.0,
                "timestamp": "2026-09-06T21:05:00+00:00",
                "temperature_f": 81,
                "dewpoint_f": 55,
                "pressure_mb": 1014.6,
                "sky_cover_oktas": 0,
                "wind_u_kt": 1.0,
                "wind_v_kt": -5.0,
                "text_description": "Clear",
                "raw_message": None,
            }
        ]
        mock_render.return_value = Path("/tmp/metar_test.png")

        result = plot_metar_observations.invoke(
            {
                "latitude": 39.77,
                "longitude": -74.89,
                "radius_km": 100.0,
            }
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["station_count"], 1)
        self.assertEqual(result["markdown_image_url"], "/tmp/metar_test.png")
        self.assertIn("regional_extent", result)


if __name__ == "__main__":
    unittest.main()
