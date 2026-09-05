"""Run the Stormy AI weather briefing agent from the command line."""

import argparse

from stormy_ai.briefing import run_briefing
from stormy_ai.config import get_settings, set_upload_to_s3


def main() -> None:
    default_location = get_settings().briefing.default_location
    parser = argparse.ArgumentParser(
        description="Generate a weather briefing with Stormy AI.",
    )
    parser.add_argument(
        "location",
        nargs="?",
        default=default_location,
        help=f"Place name for the briefing (default: {default_location})",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Write images, data, and markdown locally only; skip S3 uploads.",
    )
    args = parser.parse_args()

    if args.local:
        set_upload_to_s3(False)

    result = run_briefing(args.location)
    print(result["briefing"])
    print(f"\nWrote {result['briefing_path']}")
    if result.get("briefing_s3_uri"):
        print(f"Uploaded {result['briefing_s3_uri']}")
        if result.get("briefing_latest_s3_uri"):
            print(f"Updated {result['briefing_latest_s3_uri']}")
    elif args.local:
        print("Skipped S3 upload (--local).")
    elif result.get("briefing_s3_upload_error"):
        print(f"S3 upload failed: {result['briefing_s3_upload_error']}")


if __name__ == "__main__":
    main()
