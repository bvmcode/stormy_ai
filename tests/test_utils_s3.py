"""Tests for S3 text upload helper."""

from __future__ import annotations

import io
import unittest
from unittest.mock import MagicMock, patch

from stormy_ai.utils import upload_s3_text


class UploadS3TextTests(unittest.TestCase):
    @patch("stormy_ai.utils.s3fs.S3FileSystem")
    def test_upload_s3_text_writes_to_bucket_key(self, mock_fs_class) -> None:
        mock_fs = mock_fs_class.return_value
        buffer = io.BytesIO()
        mock_handle = MagicMock()
        mock_handle.__enter__.return_value = buffer
        mock_handle.__exit__.return_value = False
        mock_fs.open.return_value = mock_handle

        uri = upload_s3_text(
            "s3://stormy-ai-files/briefings/example.md\n",
            "s3://stormy-ai-files/latest.txt",
        )

        self.assertEqual(uri, "s3://stormy-ai-files/latest.txt")
        mock_fs.open.assert_called_once_with(
            "stormy-ai-files/latest.txt",
            "wb",
            ContentType="text/plain",
        )
        self.assertEqual(
            buffer.getvalue(),
            b"s3://stormy-ai-files/briefings/example.md\n",
        )


if __name__ == "__main__":
    unittest.main()
