# GFS guidance

`get_gfs_guidance` adapts the regional plotting ideas from the standalone
weather-model batch scripts into one agent-safe LangChain tool.

## Cycle selection

The tool accepts a location and up to four forecast lead hours. A normal
three-day briefing requests F024, F048, and F072. It finds the newest GFS
cycle for which the longest requested lead is available, then explicitly opens
every other lead from that same cycle. This prevents a briefing from silently
mixing model cycles.

## Output

For each lead, the tool returns:

- cycle, forecast hour, and valid time;
- point guidance for 2-m temperature/dewpoint, MSLP, precipitation rate,
  10-m wind, 500-hPa height/vorticity/wind, 850-hPa humidity/wind, and
  300-hPa wind;
- separate surface, 500-mb, 850-mb, and 300-mb regional PNGs;
- local paths, canonical S3 URIs, and upload-error metadata for every image.

The images show surface MSLP/thickness/precipitation/10-m wind, 500-hPa
height/vorticity/wind, 850-hPa humidity/wind, and 300-hPa height/wind. They use a
fixed North American synoptic domain, making each date/type/hour key independent
of briefing location. Local and S3 paths mirror this layout:

```text
model_plots/<YYYY-MM-DD>/<image_type>/<forecast_hour>.png
s3://stormy-ai-files/models/gfs/<YYYY-MM-DD>/<image_type>/<forecast_hour>.png
```

For readability and valid day-to-day comparison, every forecast hour uses the
same Lambert conformal map domain and fixed color thresholds. Heights are
contoured in decameters; surface pressure uses 4-hPa contours; precipitation
uses fixed operational-style rate bins; surface wind barbs use 10-m U/V;
500-hPa vorticity is lightly smoothed
and weak values are masked; 500-hPa wind barbs show the steering flow;
850-hPa moisture uses a dry-brown/moist-green palette; and 300-hPa shading
highlights winds of at least 60 kt while barbs retain the complete flow field.
Surface highs and lows are spatially filtered and limited so labels do not
obscure the underlying forecast.

`briefing.ensure_gfs_guidance_markdown` inserts the day-one, day-two, and
day-three URL for each image type if the language model omitted it. Successful
uploads use the public HTTPS object URL in sized HTML `<img>` tags
(`https://<bucket>.s3.amazonaws.com/...`); a failed upload records the error
and falls back to the local relative path.

GFS remains numerical guidance. Current observations come from NWS/MRMS/radar,
and the official public forecast and Area Forecast Discussion remain the
authoritative forecast sources.
