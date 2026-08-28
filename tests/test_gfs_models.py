from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import xarray as xr
from langchain_core.messages import ToolMessage
from pydantic import ValidationError

from stormy_ai.briefing import (
    ensure_gfs_guidance_markdown,
    extract_gfs_guidance,
)
from stormy_ai.tools.models import (
    GFS_PLOT_EXTENT,
    GFSGuidanceInput,
    _expanded_data_extent,
    _point_guidance,
    gfs_model_s3_uri,
    upload_gfs_model_image_to_s3,
)


class GFSToolTests(unittest.TestCase):
    def test_images_use_stable_north_american_domain(self):
        self.assertEqual(GFS_PLOT_EXTENT, (-127.0, -66.0, 23.0, 53.0))
        self.assertEqual(
            _expanded_data_extent(GFS_PLOT_EXTENT),
            (-142.0, -51.0, 11.0, 65.0),
        )

    def test_input_normalizes_and_limits_forecast_hours(self):
        default = GFSGuidanceInput(latitude=39.77, longitude=-74.89)
        self.assertEqual(default.forecast_hours, [24, 48, 72])

        value = GFSGuidanceInput(
            latitude=39.77,
            longitude=-74.89,
            forecast_hours=[72, 0, 24, 24],
        )
        self.assertEqual(value.forecast_hours, [0, 24, 72])

        with self.assertRaises(ValidationError):
            GFSGuidanceInput(
                latitude=39.77,
                longitude=-74.89,
                forecast_hours=[0, 1, 2, 3, 4],
            )

    def test_point_guidance_converts_units(self):
        coordinates = {
            "latitude": [40.0, 39.0],
            "longitude": [-75.0, -74.0],
        }

        def field(value):
            return xr.DataArray(
                np.full((2, 2), value),
                coords=coordinates,
                dims=("latitude", "longitude"),
            )

        fields = {
            "temperature_2m": field(293.15),
            "dewpoint_2m": field(283.15),
            "mslp": field(101325.0),
            "precip_rate": field(0.001),
            "u_10m": field(5.0),
            "v_10m": field(0.0),
            "height_500": field(5700.0),
            "vorticity_500": field(0.0002),
            "u_500": field(15.0),
            "v_500": field(0.0),
            "rh_850": field(70.0),
            "u_850": field(10.0),
            "v_850": field(0.0),
            "u_300": field(20.0),
            "v_300": field(0.0),
        }

        point = _point_guidance(fields, 39.77, -74.89)
        self.assertEqual(point["temperature_2m_c"], 20.0)
        self.assertEqual(point["precipitation_rate_mm_hr"], 3.6)
        self.assertEqual(point["wind_10m_kt"], 9.7)
        self.assertEqual(point["wind_500_kt"], 29.2)
        self.assertEqual(point["wind_850_kt"], 19.4)

    def test_s3_uri_uses_date_type_and_forecast_hour(self):
        self.assertEqual(
            gfs_model_s3_uri("2026-08-24T18:00:00Z", "850mb", 24),
            "s3://stormy-ai-files/models/gfs/2026-08-24/850mb/24.png",
        )

    def test_upload_writes_to_canonical_s3_key(self):
        with TemporaryDirectory() as directory:
            image_path = Path(directory) / "24.png"
            image_path.write_bytes(b"png")

            with patch(
                "stormy_ai.tools.models.upload_public_s3_object",
                return_value=("s3://stormy-ai-files/models/gfs/2026-08-24/500mb/24.png"),
            ) as upload:
                uri = upload_gfs_model_image_to_s3(
                    image_path,
                    "2026-08-24T18:00:00Z",
                    "500mb",
                    24,
                )

            self.assertEqual(
                uri,
                "s3://stormy-ai-files/models/gfs/2026-08-24/500mb/24.png",
            )
            upload.assert_called_once_with(
                image_path,
                "s3://stormy-ai-files/models/gfs/2026-08-24/500mb/24.png",
                content_type="image/png",
            )

    def test_markdown_insertion_is_complete_and_idempotent(self):
        guidance = {
            "model": {"cycle": "2026-08-24T18:00:00Z"},
            "images": [
                {
                    "image_type": "500mb",
                    "forecast_hour": 24,
                    "valid_time": "2026-08-25T18:00:00Z",
                    "markdown_image_url": "https://bucket.s3.amazonaws.com/500mb/24.png",
                    "include_in_markdown": True,
                },
                {
                    "image_type": "850mb",
                    "forecast_hour": 48,
                    "valid_time": "2026-08-26T18:00:00Z",
                    "markdown_image_url": "https://bucket.s3.amazonaws.com/850mb/48.png",
                    "include_in_markdown": True,
                },
                {
                    "image_type": "surface",
                    "forecast_hour": 0,
                    "valid_time": "2026-08-24T18:00:00Z",
                    "markdown_image_url": "https://bucket.s3.amazonaws.com/surface/0.png",
                    "include_in_markdown": False,
                },
            ],
        }
        message = ToolMessage(
            content=json.dumps(guidance),
            tool_call_id="gfs-1",
            name="get_gfs_guidance",
        )
        self.assertEqual(extract_gfs_guidance([message]), guidance)

        briefing = "## GFS Guidance\n\nSummary.\n\n## HRRR Analysis\n\nHRRR."
        rendered = ensure_gfs_guidance_markdown(briefing, guidance)
        self.assertEqual(
            rendered.count("https://bucket.s3.amazonaws.com/500mb/24.png"),
            1,
        )
        self.assertEqual(
            rendered.count("https://bucket.s3.amazonaws.com/850mb/48.png"),
            1,
        )
        self.assertIn('width="720"', rendered)
        self.assertIn(
            '<img src="https://bucket.s3.amazonaws.com/500mb/24.png"',
            rendered,
        )
        self.assertNotIn("https://bucket.s3.amazonaws.com/surface/0.png", rendered)
        self.assertLess(
            rendered.index("https://bucket.s3.amazonaws.com/850mb/48.png"),
            rendered.index("## HRRR Analysis"),
        )
        self.assertEqual(
            ensure_gfs_guidance_markdown(rendered, guidance),
            rendered,
        )


if __name__ == "__main__":
    unittest.main()
