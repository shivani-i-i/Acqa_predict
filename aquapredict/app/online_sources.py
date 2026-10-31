"""Adapters for public, provenance-tracked water-quality sources."""
from __future__ import annotations

import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen


USGS_IV_URL = "https://waterservices.usgs.gov/nwis/iv/"
USGS_SOURCE_URL = "https://waterdata.usgs.gov/"
DEFAULT_USGS_SITE = "01646500"
USGS_PARAMETERS = {
    "00010": "water_temperature",
    "00095": "specific_conductance",
}


class OnlineSourceError(RuntimeError):
    """Raised when an external water-quality source cannot be read."""


def fetch_usgs_observations(site: str = DEFAULT_USGS_SITE, timeout: int = 15) -> dict:
    """Fetch the latest USGS observations and preserve source metadata.

    These are environmental measurements, not household purifier telemetry.
    """
    site = str(site).strip()
    if not site.isdigit():
        raise OnlineSourceError("USGS site must be a numeric station id")
    query = urlencode({
        "format": "json",
        "sites": site,
        "parameterCd": ",".join(USGS_PARAMETERS),
        "siteStatus": "all",
    })
    request = Request(
        f"{USGS_IV_URL}?{query}",
        headers={"User-Agent": "AquaPredict/0.1 (public water-quality dashboard)"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except Exception as exc:
        raise OnlineSourceError(f"USGS request failed: {exc}") from exc

    series = payload.get("value", {}).get("timeSeries", [])
    observations = []
    site_name = site
    location = {}
    for item in series:
        source_info = item.get("sourceInfo", {})
        site_name = source_info.get("siteName") or site_name
        location = source_info.get("geoLocation", {}).get("geogLocation", {}) or location
        variable = item.get("variable", {})
        code = (variable.get("variableCode") or [{}])[0].get("value")
        for values in item.get("values", []):
            for value in values.get("value", []):
                raw = value.get("value")
                if raw in (None, "", "-999999.0"):
                    continue
                try:
                    numeric = float(raw)
                except (TypeError, ValueError):
                    continue
                observations.append({
                    "site": site,
                    "site_name": site_name,
                    "parameter_code": code,
                    "parameter": USGS_PARAMETERS.get(code, variable.get("variableName", code)),
                    "value": numeric,
                    "unit": (variable.get("unit") or {}).get("unitCode"),
                    "timestamp": value.get("dateTime"),
                    "qualifiers": value.get("qualifiers", []),
                    "source_url": USGS_SOURCE_URL,
                    "api_url": f"{USGS_IV_URL}?{query}",
                })
    if not observations:
        raise OnlineSourceError(f"USGS returned no observations for site {site}")
    return {
        "source": "USGS National Water Information System",
        "source_url": USGS_SOURCE_URL,
        "api_url": f"{USGS_IV_URL}?{query}",
        "site": site,
        "site_name": site_name,
        "latitude": location.get("latitude"),
        "longitude": location.get("longitude"),
        "retrieved_at": observations[-1]["timestamp"],
        "observations": observations,
        "limitations": [
            "Environmental station data, not household purifier telemetry.",
            "Specific conductance is not TDS and is not converted here.",
            "Provisional observations may be revised by USGS.",
            "Cannot produce filter-life labels or purifier fault labels.",
        ],
    }
