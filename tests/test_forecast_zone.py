"""Tests for NWS forecast-zone map caching and briefing embeds."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from langchain_core.messages import ToolMessage

from stormy_ai.briefing import (
    ensure_forecast_zone_markdown,
    extract_forecast_zone_info,
)
from stormy_ai.tools.nws import (
    NwsApi,
    _forecast_zone_extent,
    _geometry_bounds,
    _plot_forecast_zone,
    _select_place_labels,
    _zone_id_from_url,
    forecast_zone_s3_uri,
)


SAMPLE_GEOMETRY = {
    "type": "Polygon",
    "coordinates": [
        [
            [-75.05, 39.95],
            [-74.90, 39.95],
            [-74.90, 39.80],
            [-75.05, 39.80],
            [-75.05, 39.95],
        ]
    ],
}


class ForecastZoneHelperTests(unittest.TestCase):
    def test_forecast_zone_s3_uri(self) -> None:
        self.assertEqual(
            forecast_zone_s3_uri("NJZ018"),
            "s3://stormy-ai-files/forecast_zones/NJZ018.png",
        )

    def test_zone_id_from_url(self) -> None:
        self.assertEqual(
            _zone_id_from_url("https://api.weather.gov/zones/forecast/NJZ018"),
            "NJZ018",
        )

    def test_geometry_bounds(self) -> None:
        self.assertEqual(
            _geometry_bounds(SAMPLE_GEOMETRY),
            (-75.05, -74.90, 39.80, 39.95),
        )

    def test_forecast_zone_extent_zooms_out(self) -> None:
        extent = _forecast_zone_extent((-75.05, -74.90, 39.80, 39.95))
        west, east, south, north = extent
        self.assertAlmostEqual(east - west, 1.65, places=5)
        self.assertAlmostEqual(north - south, 1.65, places=5)
        self.assertLess(west, -75.05)
        self.assertGreater(east, -74.90)

    def test_select_place_labels_keeps_separation(self) -> None:
        labels = _select_place_labels(
            [
                ("Philadelphia", -75.16, 39.95, 2, 1500000),
                ("Camden", -75.12, 39.93, 4, 70000),
                ("Trenton", -74.76, 40.22, 3, 80000),
            ],
            max_labels=3,
            min_separation_deg=0.2,
        )
        names = [name for name, _, _ in labels]
        self.assertEqual(names[0], "Philadelphia")
        self.assertIn("Trenton", names)
        self.assertNotIn("Camden", names)

    def test_plot_forecast_zone_writes_png(self) -> None:
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "NJZ018.png"
            _plot_forecast_zone(
                SAMPLE_GEOMETRY,
                zone_id="NJZ018",
                zone_name="Camden",
                output_path=output,
            )
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 1000)

    def test_ensure_forecast_zone_markdown_uses_compact_width(self) -> None:
        text = "## Headline\n\nDry day.\n\n## Active Alerts\n\nNone.\n"
        image_url = (
            "https://stormy-ai-files.s3.amazonaws.com/forecast_zones/NJZ018.png"
        )
        result = ensure_forecast_zone_markdown(
            text,
            image_url,
            zone_id="NJZ018",
            zone_name="Camden",
        )
        self.assertIn('width="480"', result)


class ForecastZoneCacheTests(unittest.TestCase):
    @patch("stormy_ai.tools.nws.upload_public_s3_object")
    @patch("stormy_ai.tools.nws._s3_object_exists", return_value=True)
    def test_ensure_uses_cached_s3_object(
        self,
        _mock_exists,
        mock_upload,
    ) -> None:
        api = NwsApi()
        api._get = MagicMock(
            return_value={
                "geometry": SAMPLE_GEOMETRY,
                "properties": {"id": "NJZ018", "name": "Camden", "state": "NJ"},
            }
        )

        result = api.ensure_forecast_zone_image(
            "https://api.weather.gov/zones/forecast/NJZ018",
            latitude=39.77,
            longitude=-74.89,
        )

        self.assertEqual(result["status"], "success")
        self.assertTrue(result["cached"])
        self.assertEqual(result["zone_name"], "Camden")
        self.assertEqual(
            result["markdown_image_url"],
            "https://stormy-ai-files.s3.amazonaws.com/forecast_zones/NJZ018.png",
        )
        mock_upload.assert_not_called()

    @patch("stormy_ai.tools.nws.upload_public_s3_object")
    @patch("stormy_ai.tools.nws._s3_object_exists", return_value=False)
    def test_ensure_plots_and_uploads_when_missing(
        self,
        _mock_exists,
        mock_upload,
    ) -> None:
        mock_upload.side_effect = lambda path, s3_uri, content_type=None: s3_uri
        api = NwsApi()
        api._get = MagicMock(
            return_value={
                "geometry": SAMPLE_GEOMETRY,
                "properties": {"id": "NJZ018", "name": "Camden", "state": "NJ"},
            }
        )

        result = api.ensure_forecast_zone_image(
            "https://api.weather.gov/zones/forecast/NJZ018",
            latitude=39.77,
            longitude=-74.89,
        )

        self.assertEqual(result["status"], "success")
        self.assertFalse(result["cached"])
        mock_upload.assert_called_once()
        self.assertEqual(
            mock_upload.call_args.kwargs.get("content_type")
            or mock_upload.call_args[1].get("content_type"),
            "image/png",
        )


class ForecastZoneBriefingTests(unittest.TestCase):
    def test_ensure_forecast_zone_markdown_inserts_after_headline(self) -> None:
        text = "## Headline\n\nDry day.\n\n## Active Alerts\n\nNone.\n"
        image_url = (
            "https://stormy-ai-files.s3.amazonaws.com/forecast_zones/NJZ018.png"
        )
        result = ensure_forecast_zone_markdown(
            text,
            image_url,
            zone_id="NJZ018",
            zone_name="Camden",
        )
        self.assertIn("## Forecast Area", result)
        self.assertIn(image_url, result)
        self.assertLess(
            result.index("## Forecast Area"),
            result.index("## Active Alerts"),
        )
        self.assertGreater(
            result.index("## Forecast Area"),
            result.index("## Headline"),
        )

    def test_extract_forecast_zone_info(self) -> None:
        payload = {
            "status": "success",
            "forecast": "Sunny",
            "forecast_zone": {"id": "NJZ018", "name": "Camden"},
            "s3_uri": "s3://stormy-ai-files/forecast_zones/NJZ018.png",
            "markdown_image_url": (
                "https://stormy-ai-files.s3.amazonaws.com/forecast_zones/NJZ018.png"
            ),
        }
        messages = [
            ToolMessage(
                content=json.dumps(payload),
                name="get_forecast",
                tool_call_id="call-1",
            )
        ]
        info = extract_forecast_zone_info(messages)
        self.assertEqual(info["forecast_zone_id"], "NJZ018")
        self.assertEqual(info["forecast_zone_name"], "Camden")
        self.assertEqual(
            info["forecast_zone_image_url"],
            payload["markdown_image_url"],
        )


if __name__ == "__main__":
    unittest.main()
