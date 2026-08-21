"""Run the Stormy AI weather briefing agent from the command line."""

import argparse

from stormy_ai.briefing import DEFAULT_LOCATION, run_briefing


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a weather briefing with Stormy AI.",
    )
    parser.add_argument(
        "location",
        nargs="?",
        default=DEFAULT_LOCATION,
        help=f"Place name for the briefing (default: {DEFAULT_LOCATION})",
    )
    args = parser.parse_args()

    result = run_briefing(args.location)
    print(result["briefing"])
    print(f"\nWrote {result['briefing_path']}")


if __name__ == "__main__":
    main()
