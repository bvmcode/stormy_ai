"""Flask web app for Stormy AI weather briefings."""

from __future__ import annotations

from pathlib import Path

import markdown
from flask import Flask, abort, render_template, request, send_from_directory, url_for
from werkzeug.exceptions import BadRequest

from stormy_ai.briefing import (
    RADAR_PLOT_DIR,
    format_location,
    run_briefing,
)

app = Flask(__name__)


def _render_briefing_markdown(text: str) -> str:
    return markdown.markdown(
        text,
        extensions=["extra", "nl2br", "sane_lists"],
    )


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/briefing", methods=["POST"])
def briefing():
    city = request.form.get("city", "")
    state = request.form.get("state", "")

    try:
        location = format_location(city, state)
    except ValueError as exc:
        raise BadRequest(str(exc)) from exc

    try:
        result = run_briefing(location)
    except Exception as exc:
        return render_template(
            "briefing.html",
            error=str(exc),
            location=location,
        )

    radar_plot_url = None
    radar_plot_path = result["radar_plot_path"]

    if radar_plot_path and radar_plot_path.is_file():
        radar_plot_url = url_for(
            "radar_plot",
            filename=radar_plot_path.name,
        )

    return render_template(
        "briefing.html",
        location=result["location"],
        briefing_html=_render_briefing_markdown(result["briefing"]),
        radar_plot_url=radar_plot_url,
    )


@app.route("/radar/<path:filename>")
def radar_plot(filename: str):
    safe_name = Path(filename).name
    plot_path = RADAR_PLOT_DIR / safe_name

    if not plot_path.is_file():
        abort(404)

    return send_from_directory(RADAR_PLOT_DIR.resolve(), safe_name)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
