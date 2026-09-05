from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import requests
import s3fs
from langchain_core.tools import tool
from matplotlib import patheffects

from stormy_ai.config import get_settings, s3_uploads_enabled
from stormy_ai.utils import s3_uri_to_https_url, upload_public_s3_object

FORECAST_ZONE_S3_BUCKET = os.environ.get(
    "FORECAST_ZONE_S3_BUCKET",
    os.environ.get("BRIEFING_S3_BUCKET", "stormy-ai-files"),
)
FORECAST_ZONE_S3_PREFIX = os.environ.get(
    "FORECAST_ZONE_S3_PREFIX",
    "forecast_zones",
).strip("/")

MAP_CRS = ccrs.PlateCarree()
_LABEL_HALO = [patheffects.withStroke(linewidth=2.4, foreground="white")]
# Keep zone maps compact in briefings; radar/GFS stay at the default width.
FORECAST_ZONE_IMAGE_WIDTH = 480
FORECAST_ZONE_MIN_VIEW_DEG = 1.65
FORECAST_ZONE_VIEW_SCALE = 2.75
FORECAST_ZONE_MAX_LABELS = 8

COMPASS_POINTS = [
    "N",
    "NNE",
    "NE",
    "ENE",
    "E",
    "ESE",
    "SE",
    "SSE",
    "S",
    "SSW",
    "SW",
    "WSW",
    "W",
    "WNW",
    "NW",
    "NNW",
]


def _qv_value(quantity) -> float | None:
    """Extract a numeric value from an NWS QuantitativeValue object."""
    if not isinstance(quantity, dict):
        return None
    value = quantity.get("value")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _c_to_f(celsius: float) -> float:
    return celsius * 9.0 / 5.0 + 32.0


def _kmh_to_mph(km_per_hour: float) -> float:
    return km_per_hour * 0.621371


def _pa_to_inhg(pascals: float) -> float:
    return pascals / 3386.389


def _m_to_miles(meters: float) -> float:
    return meters / 1609.344


def _m_to_feet(meters: float) -> float:
    return meters * 3.28084


def _mm_to_inches(millimeters: float) -> float:
    return millimeters / 25.4


def _degrees_to_compass(degrees: float) -> str:
    index = int((degrees + 11.25) / 22.5) % 16
    return COMPASS_POINTS[index]


def _format_temp(quantity) -> str:
    celsius = _qv_value(quantity)
    if celsius is None:
        return "Not reported"
    return f"{_c_to_f(celsius):.0f}°F ({celsius:.1f}°C)"


def _format_percent(quantity) -> str:
    value = _qv_value(quantity)
    if value is None:
        return "Not reported"
    return f"{value:.0f}%"


def _format_precip(quantity) -> str:
    millimeters = _qv_value(quantity)
    if millimeters is None:
        return "Not reported"
    return f"{_mm_to_inches(millimeters):.2f} in ({millimeters:.1f} mm)"


def _format_distance_miles(meters: float | None) -> str:
    if meters is None:
        return "unknown distance"
    return f"{_m_to_miles(meters):.1f} miles"


def _format_local_timestamp(iso_time: str | None) -> str:
    if not iso_time:
        return "Unknown time"
    try:
        parsed = datetime.fromisoformat(iso_time)
    except ValueError:
        return iso_time
    stamp = parsed.strftime("%Y-%m-%d %H:%M")
    offset = parsed.strftime("%z")
    if not offset:
        return stamp
    return f"{stamp} {offset[:3]}:{offset[3:]}"


def _pop_percent(period: dict) -> str:
    value = _qv_value(period.get("probabilityOfPrecipitation"))
    if value is None:
        return "Not reported"
    return f"{value:.0f}%"


def forecast_zone_s3_uri(zone_id: str) -> str:
    """Build the canonical S3 URI for a cached forecast-zone PNG."""

    safe_zone = re.sub(r"[^A-Za-z0-9_-]+", "_", zone_id.strip()) or "unknown"
    key = f"{FORECAST_ZONE_S3_PREFIX}/{safe_zone}.png"
    return f"s3://{FORECAST_ZONE_S3_BUCKET}/{key}"


def forecast_zone_local_path(zone_id: str) -> Path:
    """Local cache path for a forecast-zone PNG."""

    safe_zone = re.sub(r"[^A-Za-z0-9_-]+", "_", zone_id.strip()) or "unknown"
    plot_dir = Path(get_settings().paths.forecast_zone_plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir / f"{safe_zone}.png"


def _zone_id_from_url(forecast_zone_url: str | None) -> str | None:
    if not forecast_zone_url:
        return None
    match = re.search(r"/zones/forecast/([^/?#]+)", forecast_zone_url)
    if match:
        return match.group(1)
    return forecast_zone_url.rstrip("/").split("/")[-1] or None


def _iter_polygon_rings(geometry: dict):
    """Yield exterior+holes ring lists from a GeoJSON Polygon or MultiPolygon."""

    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates") or []
    if geometry_type == "Polygon":
        yield coordinates
    elif geometry_type == "MultiPolygon":
        for polygon in coordinates:
            yield polygon


def _geometry_bounds(geometry: dict) -> tuple[float, float, float, float] | None:
    lons: list[float] = []
    lats: list[float] = []
    for rings in _iter_polygon_rings(geometry):
        if not rings:
            continue
        for lon, lat, *_ in rings[0]:
            lons.append(float(lon))
            lats.append(float(lat))
    if not lons or not lats:
        return None
    return min(lons), max(lons), min(lats), max(lats)


def _forecast_zone_extent(
    bounds: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Build a zoomed-out map frame so the zone has nearby geographic context."""

    min_lon, max_lon, min_lat, max_lat = bounds
    center_lon = (min_lon + max_lon) / 2.0
    center_lat = (min_lat + max_lat) / 2.0
    span_lon = max(max_lon - min_lon, 0.2)
    span_lat = max(max_lat - min_lat, 0.2)
    view = max(
        span_lon * FORECAST_ZONE_VIEW_SCALE,
        span_lat * FORECAST_ZONE_VIEW_SCALE,
        FORECAST_ZONE_MIN_VIEW_DEG,
    )
    half = view / 2.0
    return (
        center_lon - half,
        center_lon + half,
        center_lat - half,
        center_lat + half,
    )


def _place_label_candidates(
    extent: tuple[float, float, float, float],
) -> list[tuple[str, float, float, int, int]]:
    """Return ranked (name, lon, lat, scalerank, population) places in extent."""

    west, east, south, north = extent
    try:
        shapefile = shpreader.natural_earth(
            resolution="10m",
            category="cultural",
            name="populated_places",
        )
        records = list(shpreader.Reader(shapefile).records())
    except Exception:
        return []

    candidates: list[tuple[str, float, float, int, int]] = []
    for record in records:
        geometry = record.geometry
        if geometry is None:
            continue
        lon = float(geometry.x)
        lat = float(geometry.y)
        if not (west < lon < east and south < lat < north):
            continue
        attrs = record.attributes or {}
        name = str(attrs.get("NAME") or attrs.get("name") or "").strip()
        if not name:
            continue
        try:
            rank = int(attrs.get("SCALERANK") or 10)
        except (TypeError, ValueError):
            rank = 10
        try:
            population = int(attrs.get("POP_MAX") or attrs.get("POP_MIN") or 0)
        except (TypeError, ValueError):
            population = 0
        # Skip tiny hamlets that only clutter a regional map.
        if rank > 8 and population < 25000:
            continue
        candidates.append((name, lon, lat, rank, population))

    candidates.sort(key=lambda item: (item[3], -item[4], item[0]))
    return candidates


def _select_place_labels(
    candidates: list[tuple[str, float, float, int, int]],
    *,
    max_labels: int = FORECAST_ZONE_MAX_LABELS,
    min_separation_deg: float = 0.22,
) -> list[tuple[str, float, float]]:
    """Greedily keep important places without overlapping labels."""

    selected: list[tuple[str, float, float]] = []
    for name, lon, lat, _rank, _population in candidates:
        if any(
            abs(lon - other_lon) < min_separation_deg
            and abs(lat - other_lat) < min_separation_deg
            for _, other_lon, other_lat in selected
        ):
            continue
        selected.append((name, lon, lat))
        if len(selected) >= max_labels:
            break
    return selected


def _annotate_places(
    axis,
    extent: tuple[float, float, float, float],
) -> None:
    labels = _select_place_labels(_place_label_candidates(extent))
    for name, lon, lat in labels:
        axis.plot(
            lon,
            lat,
            marker="o",
            markersize=3.2,
            color="#0f172a",
            markeredgecolor="white",
            markeredgewidth=0.6,
            transform=MAP_CRS,
            zorder=7,
        )
        text = axis.annotate(
            name,
            xy=(lon, lat),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7.5,
            color="#0f172a",
            fontweight="bold",
            transform=MAP_CRS,
            zorder=8,
        )
        text.set_path_effects(_LABEL_HALO)


def _plot_forecast_zone(
    geometry: dict,
    *,
    zone_id: str,
    zone_name: str,
    output_path: Path,
) -> Path:
    """Render a compact, zoomed-out forecast-zone map with place labels."""

    bounds = _geometry_bounds(geometry)
    if bounds is None:
        raise ValueError(f"Forecast zone {zone_id} has no plotable coordinates.")

    extent = _forecast_zone_extent(bounds)

    figure = plt.figure(figsize=(5.2, 4.9), facecolor="white")
    axis = figure.add_axes([0.06, 0.08, 0.88, 0.82], projection=MAP_CRS)
    axis.set_extent(extent, crs=MAP_CRS)
    axis.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f4f1e8", zorder=0)
    axis.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#dbeafe", zorder=0)
    axis.add_feature(
        cfeature.LAKES.with_scale("10m"),
        facecolor="#bfdbfe",
        edgecolor="#64748b",
        linewidth=0.35,
        zorder=1,
    )
    axis.add_feature(
        cfeature.RIVERS.with_scale("10m"),
        edgecolor="#60a5fa",
        linewidth=0.55,
        zorder=1,
    )
    axis.add_feature(
        cfeature.COASTLINE.with_scale("10m"),
        edgecolor="#1e293b",
        linewidth=0.7,
        zorder=2,
    )
    axis.add_feature(
        cfeature.STATES.with_scale("10m"),
        edgecolor="#475569",
        linewidth=0.7,
        zorder=2,
    )
    axis.add_feature(
        cfeature.BORDERS.with_scale("10m"),
        edgecolor="#334155",
        linewidth=0.85,
        zorder=2,
    )
    try:
        counties = cfeature.NaturalEarthFeature(
            category="cultural",
            name="admin_2_counties",
            scale="10m",
            facecolor="none",
            edgecolor="#94a3b8",
            linewidth=0.25,
        )
        axis.add_feature(counties, zorder=2)
    except Exception:
        pass

    for rings in _iter_polygon_rings(geometry):
        if not rings:
            continue
        exterior = rings[0]
        lons = [float(point[0]) for point in exterior]
        lats = [float(point[1]) for point in exterior]
        axis.fill(
            lons,
            lats,
            transform=MAP_CRS,
            facecolor="#2563eb",
            alpha=0.38,
            edgecolor="#1d4ed8",
            linewidth=1.5,
            zorder=4,
        )
        for hole in rings[1:]:
            hole_lons = [float(point[0]) for point in hole]
            hole_lats = [float(point[1]) for point in hole]
            axis.fill(
                hole_lons,
                hole_lats,
                transform=MAP_CRS,
                facecolor="#f4f1e8",
                edgecolor="#1d4ed8",
                linewidth=0.7,
                zorder=5,
            )

    _annotate_places(axis, extent)

    gridlines = axis.gridlines(
        draw_labels=True,
        crs=MAP_CRS,
        linewidth=0.3,
        linestyle="--",
        color="#94a3b8",
        alpha=0.45,
        zorder=3,
    )
    gridlines.top_labels = False
    gridlines.right_labels = False
    gridlines.xlabel_style = {"size": 7, "color": "#64748b"}
    gridlines.ylabel_style = {"size": 7, "color": "#64748b"}

    title = f"NWS Forecast Zone {zone_id}"
    if zone_name:
        title = f"{title} — {zone_name}"
    axis.set_title(title, fontsize=10, pad=8)
    figure.text(
        0.5,
        0.015,
        "Blue shading = area covered by this official NWS forecast",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#475569",
    )
    figure.savefig(
        output_path,
        dpi=110,
        facecolor=figure.get_facecolor(),
        edgecolor="none",
    )
    plt.close(figure)
    return output_path


def _s3_object_exists(s3_uri: str) -> bool:
    filesystem = s3fs.S3FileSystem(anon=False)
    return bool(filesystem.exists(s3_uri.removeprefix("s3://")))


class NwsApi:
    """Client for the National Weather Service API."""

    # The NWS API needs a User-Agent that identifies your app.
    # No API key is required.
    BASE_URL = "https://api.weather.gov"
    USER_AGENT = "stormy-ai/0.1 (stormy-ai@example.com)"

    def _get(
        self,
        url: str,
        accept: str = "application/geo+json",
    ) -> dict | None:
        """GET JSON from the NWS API, or return None on failure."""
        try:
            response = requests.get(
                url,
                headers={
                    "User-Agent": self.USER_AGENT,
                    "Accept": accept,
                },
                timeout=30,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            return None

    def _get_points(self, latitude: float, longitude: float) -> dict | None:
        """Translate lat/lon into NWS point metadata."""
        points_url = f"{self.BASE_URL}/points/{latitude:.4f},{longitude:.4f}"
        return self._get(points_url)

    def ensure_forecast_zone_image(
        self,
        forecast_zone_url: str,
        latitude: float,
        longitude: float,
    ) -> dict:
        """
        Return a forecast-zone map image, generating it only when missing.

        Cached objects live at ``s3://<bucket>/forecast_zones/<zone>.png`` when
        S3 uploads are enabled, and under ``paths.forecast_zone_plot_dir`` locally.
        """

        zone_id = _zone_id_from_url(forecast_zone_url)
        if not zone_id:
            return {
                "status": "error",
                "error": "Unable to determine the NWS forecast zone id.",
            }

        s3_uri = forecast_zone_s3_uri(zone_id)
        local_path = forecast_zone_local_path(zone_id)
        upload_enabled = s3_uploads_enabled()
        zone_name = ""
        zone_state = ""

        def _zone_metadata() -> tuple[str, str]:
            zone_data = self._get(forecast_zone_url)
            if not zone_data:
                return "", ""
            properties = zone_data.get("properties") or {}
            return (
                str(properties.get("name") or ""),
                str(properties.get("state") or ""),
            )

        if upload_enabled:
            try:
                if _s3_object_exists(s3_uri):
                    zone_name, zone_state = _zone_metadata()
                    https_url = s3_uri_to_https_url(s3_uri)
                    return {
                        "status": "success",
                        "zone_id": zone_id,
                        "zone_name": zone_name,
                        "zone_state": zone_state,
                        "image_path": (
                            str(local_path.resolve()) if local_path.is_file() else None
                        ),
                        "s3_uri": s3_uri,
                        "https_url": https_url,
                        "markdown_image_url": https_url,
                        "cached": True,
                        "s3_upload_error": None,
                    }
            except Exception:
                # Fall through and regenerate when the existence check fails
                # (for example missing credentials in a local scratch run).
                pass
        elif local_path.is_file():
            zone_name, zone_state = _zone_metadata()
            local_url = str(local_path.resolve())
            return {
                "status": "success",
                "zone_id": zone_id,
                "zone_name": zone_name,
                "zone_state": zone_state,
                "image_path": local_url,
                "s3_uri": None,
                "https_url": None,
                "markdown_image_url": local_url,
                "cached": True,
                "s3_upload_error": None,
            }

        zone_data = self._get(forecast_zone_url)
        if not zone_data:
            return {
                "status": "error",
                "zone_id": zone_id,
                "error": f"Unable to fetch forecast zone geometry for {zone_id}.",
            }

        properties = zone_data.get("properties") or {}
        zone_name = str(properties.get("name") or "")
        zone_state = str(properties.get("state") or "")
        geometry = zone_data.get("geometry")
        if not isinstance(geometry, dict) or not geometry.get("coordinates"):
            return {
                "status": "error",
                "zone_id": zone_id,
                "zone_name": zone_name,
                "zone_state": zone_state,
                "error": f"Forecast zone {zone_id} did not include geometry.",
            }

        try:
            _plot_forecast_zone(
                geometry,
                zone_id=zone_id,
                zone_name=zone_name,
                output_path=local_path,
            )
        except Exception as exc:
            return {
                "status": "error",
                "zone_id": zone_id,
                "zone_name": zone_name,
                "zone_state": zone_state,
                "error": f"Unable to plot forecast zone {zone_id}: {exc}",
            }

        local_url = str(local_path.resolve())
        s3_upload_error = None
        if upload_enabled:
            try:
                upload_public_s3_object(
                    local_path,
                    s3_uri,
                    content_type="image/png",
                )
            except Exception as exc:
                s3_uri = None
                s3_upload_error = str(exc)
        else:
            s3_uri = None

        https_url = s3_uri_to_https_url(s3_uri) if s3_uri else None
        markdown_image_url = https_url or local_url
        return {
            "status": "success" if markdown_image_url else "error",
            "zone_id": zone_id,
            "zone_name": zone_name,
            "zone_state": zone_state,
            "image_path": local_url,
            "s3_uri": s3_uri,
            "https_url": https_url,
            "markdown_image_url": markdown_image_url,
            "cached": False,
            "s3_upload_error": s3_upload_error,
            "error": None if markdown_image_url else (s3_upload_error or "Upload failed."),
        }

    def _format_period(self, period: dict) -> str:
        pop = _pop_percent(period)
        trend = period.get("temperatureTrend")
        trend_text = f" (trend: {trend})" if trend else ""
        return (
            f"{period['name']}:\n"
            f"Time: {period.get('startTime', 'unknown')} to "
            f"{period.get('endTime', 'unknown')}\n"
            f"Temperature: {period['temperature']}°"
            f"{period['temperatureUnit']}{trend_text}\n"
            f"Wind: {period.get('windSpeed', 'unknown')} "
            f"{period.get('windDirection') or ''}\n"
            f"Chance of precipitation: {pop}\n"
            f"Short forecast: {period.get('shortForecast', '')}\n"
            f"Forecast: {period.get('detailedForecast', '')}"
        )

    def _format_hourly_period(self, period: dict) -> str:
        dewpoint = _format_temp(period.get("dewpoint"))
        humidity = _format_percent(period.get("relativeHumidity"))
        pop = _pop_percent(period)
        start = _format_local_timestamp(period.get("startTime"))
        wind = (
            f"{period.get('windSpeed', 'unknown')} " f"{period.get('windDirection') or ''}"
        ).strip()
        return (
            f"{start} | {period.get('temperature')}°"
            f"{period.get('temperatureUnit')} | "
            f"Dewpoint {dewpoint} | RH {humidity} | "
            f"Wind {wind} | PoP {pop} | "
            f"{period.get('shortForecast', '')}"
        )

    def _format_cloud_layers(self, layers) -> str:
        if not layers:
            return "Not reported"
        formatted = []
        for layer in layers:
            amount = layer.get("amount", "unknown")
            base_m = _qv_value(layer.get("base"))
            if base_m is None:
                formatted.append(str(amount))
            else:
                formatted.append(f"{amount} at {_m_to_feet(base_m):.0f} ft")
        return "; ".join(formatted)

    def _format_present_weather(self, phenomena) -> str:
        if not phenomena:
            return "None reported"
        parts = []
        for item in phenomena:
            if not isinstance(item, dict):
                continue
            raw = item.get("rawString")
            if raw:
                parts.append(raw)
                continue
            tokens = [
                item.get("intensity"),
                item.get("modifier"),
                item.get("weather"),
            ]
            label = " ".join(token for token in tokens if token)
            if label:
                parts.append(label)
        return ", ".join(parts) if parts else "None reported"

    def _format_wind(self, observation: dict) -> str:
        speed_kmh = _qv_value(observation.get("windSpeed"))
        gust_kmh = _qv_value(observation.get("windGust"))
        direction = _qv_value(observation.get("windDirection"))

        if speed_kmh is None:
            return "Not reported"
        if speed_kmh == 0:
            return "Calm"

        speed_mph = _kmh_to_mph(speed_kmh)
        if direction is None:
            wind = f"{speed_mph:.0f} mph"
        else:
            compass = _degrees_to_compass(direction)
            wind = f"{compass} at {speed_mph:.0f} mph ({direction:.0f}°)"

        if gust_kmh:
            wind += f", gusts {_kmh_to_mph(gust_kmh):.0f} mph"

        return wind

    def _format_pressure(self, quantity, label: str) -> str:
        pascals = _qv_value(quantity)
        if pascals is None:
            return f"{label}: Not reported"
        inhg = _pa_to_inhg(pascals)
        hpa = pascals / 100.0
        return f"{label}: {inhg:.2f} inHg ({hpa:.1f} mb)"

    def _format_visibility(self, quantity) -> str:
        meters = _qv_value(quantity)
        if meters is None:
            return "Not reported"
        miles = _m_to_miles(meters)
        if miles >= 10:
            return f"{miles:.0f} miles"
        return f"{miles:.1f} miles"

    def _format_observation(
        self,
        station: dict,
        observation: dict,
    ) -> str:
        station_props = station.get("properties") or {}
        obs = observation.get("properties") or {}
        distance_m = _qv_value(station_props.get("distance"))
        elevation_m = _qv_value(station_props.get("elevation") or obs.get("elevation"))
        elevation = (
            f"{_m_to_feet(elevation_m):.0f} ft" if elevation_m is not None else "Not reported"
        )
        heat_index = _qv_value(obs.get("heatIndex"))
        wind_chill = _qv_value(obs.get("windChill"))
        apparent = []
        if heat_index is not None:
            apparent.append(f"Heat index: {_format_temp(obs.get('heatIndex'))}")
        if wind_chill is not None:
            apparent.append(f"Wind chill: {_format_temp(obs.get('windChill'))}")

        lines = [
            f"Station: {station_props.get('name', 'Unknown')} "
            f"({station_props.get('stationIdentifier', 'unknown')})",
            f"Distance from requested point: {_format_distance_miles(distance_m)}",
            f"Elevation: {elevation}",
            f"Observed at: {_format_local_timestamp(obs.get('timestamp'))}",
            f"Conditions: {obs.get('textDescription') or 'Not reported'}",
            "Present weather: " f"{self._format_present_weather(obs.get('presentWeather'))}",
            f"Temperature: {_format_temp(obs.get('temperature'))}",
            f"Dewpoint: {_format_temp(obs.get('dewpoint'))}",
            f"Relative humidity: {_format_percent(obs.get('relativeHumidity'))}",
            f"Wind: {self._format_wind(obs)}",
            self._format_pressure(
                obs.get("barometricPressure"),
                "Barometric pressure",
            ),
            self._format_pressure(
                obs.get("seaLevelPressure"),
                "Sea level pressure",
            ),
            f"Visibility: {self._format_visibility(obs.get('visibility'))}",
            f"Clouds: {self._format_cloud_layers(obs.get('cloudLayers'))}",
            "Precipitation last hour: " f"{_format_precip(obs.get('precipitationLastHour'))}",
            "Precipitation last 3 hours: " f"{_format_precip(obs.get('precipitationLast3Hours'))}",
            "Precipitation last 6 hours: " f"{_format_precip(obs.get('precipitationLast6Hours'))}",
            "Max temperature last 24 hours: "
            f"{_format_temp(obs.get('maxTemperatureLast24Hours'))}",
            "Min temperature last 24 hours: "
            f"{_format_temp(obs.get('minTemperatureLast24Hours'))}",
        ]
        if apparent:
            lines.extend(apparent)
        raw_message = obs.get("rawMessage")
        if raw_message:
            lines.append(f"Raw METAR: {raw_message}")
        return "\n".join(lines)

    def get_forecast(self, latitude: float, longitude: float) -> dict:
        """Get the weather forecast for a US location from the National Weather Service.

        Returns 12-hour forecast periods covering today through the next two
        days, plus an hourly forecast for the next 72 hours. Also returns a
        cached map of the NWS forecast zone where the official forecast is
        valid (``markdown_image_url``).

        Args:
            latitude: Latitude of the location (e.g. 40.7128 for New York).
            longitude: Longitude of the location (e.g. -74.0060 for New York).
        """
        points_data = self._get_points(latitude, longitude)
        if not points_data:
            return {
                "status": "error",
                "error": "Unable to fetch forecast data for this location.",
            }

        properties = points_data.get("properties") or {}
        forecast_url = properties.get("forecast")
        hourly_url = properties.get("forecastHourly")
        forecast_zone_url = properties.get("forecastZone")
        if not forecast_url:
            return {
                "status": "error",
                "error": "Unable to fetch forecast data for this location.",
            }

        forecast_data = self._get(forecast_url)
        if not forecast_data:
            return {
                "status": "error",
                "error": "Unable to fetch detailed forecast.",
            }

        periods = (forecast_data.get("properties") or {}).get("periods") or []
        if not periods:
            return {
                "status": "error",
                "error": "No forecast periods were returned for this location.",
            }

        # About eight 12-hour periods covers today plus the next two days.
        period_lines = [self._format_period(period) for period in periods[:8]]
        sections = [
            "12-hour forecast periods (today through the next two days):",
            "\n\n".join(period_lines),
        ]

        if hourly_url:
            hourly_data = self._get(hourly_url)
            hourly_periods = ((hourly_data or {}).get("properties") or {}).get("periods") or []
            if hourly_periods:
                hourly_lines = [
                    self._format_hourly_period(period) for period in hourly_periods[:72]
                ]
                sections.append("Hourly forecast (next 72 hours):\n" + "\n".join(hourly_lines))

        forecast_text = "\n\n".join(sections)
        zone_image: dict = {"status": "skipped"}
        if forecast_zone_url:
            zone_image = self.ensure_forecast_zone_image(
                forecast_zone_url,
                latitude,
                longitude,
            )
            zone_id = zone_image.get("zone_id") or _zone_id_from_url(forecast_zone_url)
            zone_name = zone_image.get("zone_name") or ""
            zone_label = (
                f"{zone_id} ({zone_name})" if zone_id and zone_name else (zone_id or "unknown")
            )
            forecast_text = (
                f"Forecast zone: {zone_label}\n"
                f"markdown_image_url: {zone_image.get('markdown_image_url') or 'unavailable'}\n\n"
                f"{forecast_text}"
            )

        return {
            "status": "success",
            "forecast": forecast_text,
            "forecast_zone": {
                "id": zone_image.get("zone_id"),
                "name": zone_image.get("zone_name"),
                "state": zone_image.get("zone_state"),
                "url": forecast_zone_url,
                "cached": zone_image.get("cached"),
                "status": zone_image.get("status"),
                "error": zone_image.get("error") or zone_image.get("s3_upload_error"),
            },
            "s3_uri": zone_image.get("s3_uri"),
            "https_url": zone_image.get("https_url"),
            "markdown_image_url": zone_image.get("markdown_image_url"),
            "image_path": zone_image.get("image_path"),
        }

    def get_alerts(self, latitude: float, longitude: float) -> str:
        """Get active weather alerts for a US location from NWS.

        Args:
            latitude: Latitude of the location (e.g. 40.7128 for New York).
            longitude: Longitude of the location (e.g. -74.0060 for New York).
        """
        alerts_url = f"{self.BASE_URL}/alerts/active?point={latitude:.4f},{longitude:.4f}"
        alerts_data = self._get(alerts_url)
        if not alerts_data:
            return "Unable to fetch alerts for this location."

        features = alerts_data.get("features", [])
        if not features:
            return "No active alerts for this location."

        lines = []
        for feature in features:
            props = feature["properties"]
            lines.append(
                f"Event: {props['event']}\n"
                f"Severity: {props['severity']}\n"
                f"Area: {props['areaDesc']}\n"
                f"Effective: {props['effective']}\n"
                f"Expires: {props['expires']}\n"
                f"Description: {props['description']}\n"
                f"Instructions: {props.get('instruction') or 'None provided'}"
            )
        return "\n\n---\n\n".join(lines)

    def current_conditions(self, latitude: float, longitude: float) -> str:
        """Get the latest official NWS surface observations near a US location.

        Uses the closest METAR/ASOS station. This is the primary source for
        current temperature, dewpoint, wind, pressure, visibility, humidity,
        and sky cover. It is an observation, not a forecast.

        Args:
            latitude: Latitude of the location (e.g. 40.7128 for New York).
            longitude: Longitude of the location (e.g. -74.0060 for New York).
        """
        points_data = self._get_points(latitude, longitude)
        if not points_data:
            return "Unable to fetch current conditions for this location."

        stations_url = (points_data.get("properties") or {}).get("observationStations")
        if not stations_url:
            return "No observation stations were found for this location."

        stations_data = self._get(stations_url)
        stations = (stations_data or {}).get("features") or []
        if not stations:
            return "No observation stations were found for this location."

        tried = []
        for station in stations[:5]:
            station_id = (station.get("properties") or {}).get("stationIdentifier")
            if not station_id:
                continue
            tried.append(station_id)
            observation = self._get(f"{self.BASE_URL}/stations/{station_id}/observations/latest")
            if not observation:
                continue
            return self._format_observation(station, observation)

        names = ", ".join(tried) if tried else "nearby stations"
        return f"Unable to fetch a usable latest observation from {names}."

    def forecast_discussion(self, latitude: float, longitude: float) -> str:
        """Get the latest NWS Area Forecast Discussion for a US location.

        The Area Forecast Discussion (AFD) is the forecast office's written
        reasoning: what has changed, key messages, short-term and longer-term
        thinking, and hazards. Use it to explain why the forecast looks the
        way it does.

        Args:
            latitude: Latitude of the location (e.g. 40.7128 for New York).
            longitude: Longitude of the location (e.g. -74.0060 for New York).
        """
        points_data = self._get_points(latitude, longitude)
        if not points_data:
            return "Unable to fetch the forecast discussion for this location."

        cwa = (points_data.get("properties") or {}).get("cwa")
        if not cwa:
            return "Unable to determine the NWS forecast office for this location."

        latest_url = f"{self.BASE_URL}/products/types/AFD/locations/{cwa}/latest"
        product = self._get(latest_url, accept="application/ld+json")

        if not product:
            listing = self._get(
                f"{self.BASE_URL}/products/types/AFD/locations/{cwa}",
                accept="application/ld+json",
            )
            graph = (listing or {}).get("@graph") or []
            if not graph:
                return f"No Area Forecast Discussion is available for office {cwa}."
            product_url = graph[0].get("@id") or (f"{self.BASE_URL}/products/{graph[0].get('id')}")
            product = self._get(product_url, accept="application/ld+json")

        if not product:
            return f"Unable to fetch the Area Forecast Discussion for office {cwa}."

        text = (product.get("productText") or "").strip()
        if not text:
            return f"The Area Forecast Discussion for office {cwa} was empty."

        office = product.get("issuingOffice") or cwa
        issued = _format_local_timestamp(product.get("issuanceTime"))
        name = product.get("productName") or "Area Forecast Discussion"
        return f"Product: {name}\n" f"Office: {office}\n" f"Issued: {issued}\n\n" f"{text}"


nws_api = NwsApi()
get_forecast = tool(nws_api.get_forecast)
get_alerts = tool(nws_api.get_alerts)
current_conditions = tool(nws_api.current_conditions)
forecast_discussion = tool(nws_api.forecast_discussion)
