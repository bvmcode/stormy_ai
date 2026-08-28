# @tool turns a normal Python function into something the
# LLM can discover and call by name.

from __future__ import annotations

from datetime import datetime

import requests
from langchain_core.tools import tool

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
    except TypeError, ValueError:
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

    def get_forecast(self, latitude: float, longitude: float) -> str:
        """Get the weather forecast for a US location from the National Weather Service.

        Returns 12-hour forecast periods covering today through the next two
        days, plus an hourly forecast for the next 72 hours.

        Args:
            latitude: Latitude of the location (e.g. 40.7128 for New York).
            longitude: Longitude of the location (e.g. -74.0060 for New York).
        """
        points_data = self._get_points(latitude, longitude)
        if not points_data:
            return "Unable to fetch forecast data for this location."

        properties = points_data.get("properties") or {}
        forecast_url = properties.get("forecast")
        hourly_url = properties.get("forecastHourly")
        if not forecast_url:
            return "Unable to fetch forecast data for this location."

        forecast_data = self._get(forecast_url)
        if not forecast_data:
            return "Unable to fetch detailed forecast."

        periods = (forecast_data.get("properties") or {}).get("periods") or []
        if not periods:
            return "No forecast periods were returned for this location."

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

        return "\n\n".join(sections)

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
