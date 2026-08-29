"""Tests for briefing S3 upload helpers."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from stormy_ai.briefing import (
    briefing_latest_s3_uri,
    briefing_s3_uri,
    update_briefing_latest_pointer,
    upload_briefing_to_s3,
)


class BriefingS3Tests(unittest.TestCase):
    def test_briefing_s3_uri_layout(self) -> None:
        when = datetime(2026, 8, 29, 16, 30, tzinfo=timezone.utc)
        uri = briefing_s3_uri(when, "08004")
        self.assertEqual(
            uri,
            "s3://stormy-ai-files/briefings/2026-08-29/08004/16_30.md",
        )

    def test_briefing_latest_s3_uri(self) -> None:
        self.assertEqual(
            briefing_latest_s3_uri(),
            "s3://stormy-ai-files/latest.txt",
        )

    @patch("stormy_ai.briefing.upload_s3_text")
    def test_update_briefing_latest_pointer(self, mock_upload) -> None:
        briefing_uri = "s3://stormy-ai-files/briefings/2026-08-29/08004/16_30.md"
        latest_uri = update_briefing_latest_pointer(briefing_uri)

        self.assertEqual(latest_uri, "s3://stormy-ai-files/latest.txt")
        mock_upload.assert_called_once_with(
            f"{briefing_uri}\n",
            latest_uri,
        )

    @patch("stormy_ai.briefing.update_briefing_latest_pointer")
    @patch("stormy_ai.briefing.upload_public_s3_object")
    def test_upload_briefing_to_s3_updates_latest_pointer(
        self,
        mock_upload_object,
        mock_update_latest,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "briefing.md"
            path.write_text("# Briefing\n", encoding="utf-8")
            when = datetime(2026, 8, 29, 16, 30, tzinfo=timezone.utc)
            expected_uri = briefing_s3_uri(when, "08004")
            mock_upload_object.return_value = expected_uri

            result = upload_briefing_to_s3(path, zip_code="08004", when=when)

        self.assertEqual(result, expected_uri)
        mock_upload_object.assert_called_once_with(
            path,
            expected_uri,
            content_type="text/markdown",
        )
        mock_update_latest.assert_called_once_with(expected_uri)


if __name__ == "__main__":
    unittest.main()
