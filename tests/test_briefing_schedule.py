"""Tests for briefing update schedule helpers."""

from __future__ import annotations

import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from stormy_ai.briefing import (
    BRIEFING_SCHEDULE_TZ,
    briefing_schedule_times,
    format_briefing_schedule_time,
)

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
    def test_midday_run_uses_noon_slot(self) -> None:
        when = _eastern(2026, 8, 29, 12, 7)
        current, next_update = briefing_schedule_times(when)

        self.assertEqual(current, _eastern(2026, 8, 29, 12))
        self.assertEqual(next_update, _eastern(2026, 8, 29, 18))

    def test_early_morning_run_uses_midnight_slot(self) -> None:
        when = _eastern(2026, 8, 29, 3, 15)
        current, next_update = briefing_schedule_times(when)

        self.assertEqual(current, _eastern(2026, 8, 29, 0))
        self.assertEqual(next_update, _eastern(2026, 8, 29, 6))

    def test_evening_run_wraps_to_midnight(self) -> None:
        when = _eastern(2026, 8, 29, 19, 45)
        current, next_update = briefing_schedule_times(when)

        self.assertEqual(current, _eastern(2026, 8, 29, 18))
        self.assertEqual(next_update, _eastern(2026, 8, 30, 0))

    def test_exact_schedule_boundary(self) -> None:
        when = _eastern(2026, 8, 29, 6, 0)
        current, next_update = briefing_schedule_times(when)

        self.assertEqual(current, _eastern(2026, 8, 29, 6))
        self.assertEqual(next_update, _eastern(2026, 8, 29, 12))

    def test_format_briefing_schedule_time(self) -> None:
        formatted = format_briefing_schedule_time(_eastern(2026, 8, 29, 18))
        self.assertEqual(formatted, "Saturday, August 29, 2026 6:00 PM Eastern")

    def test_schedule_timezone_constant(self) -> None:
        self.assertEqual(BRIEFING_SCHEDULE_TZ, EASTERN)


if __name__ == "__main__":
    unittest.main()
