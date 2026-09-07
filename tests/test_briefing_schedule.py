"""Tests for briefing update schedule helpers."""

from __future__ import annotations

import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from stormy_ai.briefing import (
    briefing_schedule_times,
    format_briefing_schedule_time,
)
from stormy_ai.config import get_settings

EASTERN = ZoneInfo("America/New_York")


def _eastern(
    year: int,
    month: int,
    day: int,
    hour: int,
    minute: int = 0,
) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=EASTERN)


class BriefingScheduleTests(unittest.TestCase):
    def test_midday_run_uses_hourly_slot(self) -> None:
        when = _eastern(2026, 8, 29, 12, 7)
        current, next_update = briefing_schedule_times(when)

        self.assertEqual(current, _eastern(2026, 8, 29, 12))
        self.assertEqual(next_update, _eastern(2026, 8, 29, 13))

    def test_overnight_run_uses_4am_slot(self) -> None:
        when = _eastern(2026, 8, 29, 5, 15)
        current, next_update = briefing_schedule_times(when)

        self.assertEqual(current, _eastern(2026, 8, 29, 4))
        self.assertEqual(next_update, _eastern(2026, 8, 29, 8))

    def test_evening_run_advances_hourly_until_8pm(self) -> None:
        when = _eastern(2026, 8, 29, 19, 45)
        current, next_update = briefing_schedule_times(when)

        self.assertEqual(current, _eastern(2026, 8, 29, 19))
        self.assertEqual(next_update, _eastern(2026, 8, 29, 20))

    def test_after_8pm_wraps_to_4am(self) -> None:
        when = _eastern(2026, 8, 29, 21, 10)
        current, next_update = briefing_schedule_times(when)

        self.assertEqual(current, _eastern(2026, 8, 29, 20))
        self.assertEqual(next_update, _eastern(2026, 8, 30, 4))

    def test_exact_schedule_boundary(self) -> None:
        when = _eastern(2026, 8, 29, 8, 0)
        current, next_update = briefing_schedule_times(when)

        self.assertEqual(current, _eastern(2026, 8, 29, 8))
        self.assertEqual(next_update, _eastern(2026, 8, 29, 9))

    def test_format_briefing_schedule_time(self) -> None:
        formatted = format_briefing_schedule_time(_eastern(2026, 8, 29, 18))
        self.assertEqual(formatted, "Saturday, August 29, 2026 6:00 PM Eastern")

    def test_schedule_timezone_constant(self) -> None:
        self.assertEqual(
            ZoneInfo(get_settings().briefing.schedule_tz),
            EASTERN,
        )

    def test_configured_schedule_hours(self) -> None:
        self.assertEqual(
            get_settings().briefing.schedule_hours,
            (4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20),
        )


if __name__ == "__main__":
    unittest.main()
