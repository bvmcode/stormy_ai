import unittest
from datetime import datetime, timedelta, timezone

import numpy as np

from stormy_ai.tools.radar import (
    _range_ring_distances_km,
    _visible_field_max,
    radar_plot_s3_uri,
)


class FakeRadar:
    def get_gate_lat_lon_alt(self, sweep):
        del sweep
        return (
            np.array([[39.5, 39.8, 40.5]]),
            np.array([[-75.0, -74.8, -73.0]]),
            np.zeros((1, 3)),
        )

    def get_field(self, sweep, field):
        del sweep, field
        return np.ma.array([[4.0, 31.0, 60.0]])


class RadarPlotHelpersTest(unittest.TestCase):
    def test_s3_uri_uses_utc_scan_time(self):
        eastern = timezone(timedelta(hours=-4))
        scan_time = datetime(2026, 8, 24, 20, 48, tzinfo=eastern)

        self.assertEqual(
            radar_plot_s3_uri(scan_time),
            "s3://stormy-ai-files/radar/2026-08-25/00_48.png",
        )

    def test_range_rings_stop_near_visible_radius(self):
        self.assertEqual(
            _range_ring_distances_km(100),
            [25.0, 50.0, 75.0, 100.0],
        )
        self.assertEqual(_range_ring_distances_km(75), [25.0, 50.0, 75.0])

    def test_visible_max_respects_bounds_and_minimum(self):
        maximum = _visible_field_max(
            FakeRadar(),
            sweep=0,
            radar_field="reflectivity",
            bounds=(-75.5, -74.0, 39.0, 40.0),
            minimum=5.0,
        )

        self.assertEqual(maximum, 31.0)


if __name__ == "__main__":
    unittest.main()
