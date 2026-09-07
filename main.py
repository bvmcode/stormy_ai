"""Run the Stormy AI weather briefing agent from the command line."""

import argparse
import sys
from time import perf_counter

from stormy_ai.briefing import run_briefing
from stormy_ai.config import get_settings, set_upload_to_s3
from stormy_ai.logging_config import configure_logging, format_kv, get_logger


def main() -> None:
    configure_logging()
    logger = get_logger(__name__)

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

    settings = get_settings()
    logger.info(
        "cli.start %s",
        format_kv(
            location=args.location,
            local=args.local,
            upload_to_s3=settings.storage.upload_to_s3,
            llm_provider=settings.llm.provider,
            llm_model=settings.llm.model,
        ),
    )
    started = perf_counter()
    try:
        result = run_briefing(args.location)
    except Exception:
        logger.exception(
            "cli.failed %s",
            format_kv(
                location=args.location,
                duration_s=perf_counter() - started,
            ),
        )
        raise SystemExit(1) from None

    logger.info(
        "cli.complete %s",
        format_kv(
            location=result.get("location"),
            duration_s=perf_counter() - started,
            briefing_path=result.get("briefing_path"),
            briefing_s3_uri=result.get("briefing_s3_uri"),
            s3_upload_error=result.get("briefing_s3_upload_error"),
        ),
    )

    print(result["briefing"])
    print(f"\nWrote {result['briefing_path']}")
    if result.get("briefing_s3_uri"):
        print(f"Uploaded {result['briefing_s3_uri']}")
        if result.get("briefing_latest_s3_uri"):
            print(f"Updated {result['briefing_latest_s3_uri']}")
    elif args.local:
        print("Skipped S3 upload (--local).")
    elif result.get("briefing_s3_upload_error"):
        print(f"S3 upload failed: {result['briefing_s3_upload_error']}", file=sys.stderr)


if __name__ == "__main__":
    main()
