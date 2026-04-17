"""
carbon_inference_engine.py
==========================
Green-Ops CI/CD Framework — Carbon Intensity Client

CHANGES (v2):
  - FIX: Original always hit UK National Grid API (api.carbonintensity.org.uk)
         which returns UK data. The pipeline targets Indian datacenters, so
         this was always returning incorrect intensity values for routing.
         Now supports an India-specific intensity fetch (via CO2Signal/Electricity
         Maps API) with the UK endpoint as a secondary fallback for normalisation.
  - FIX: No retry logic; single timeout failure caused fallback immediately.
         Added configurable retry with exponential back-off.
  - FIX: `actual_intensity` could be None but code cast it to int without guard.
         Added explicit None check before return.
  - IMPROVEMENT: Configurable via environment variables.
  - IMPROVEMENT: Returns structured result with source metadata for audit trail.
"""

import json
import logging
import os
import time
import urllib.request
from typing import Optional

logger = logging.getLogger("GreenOps.CarbonInference")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Primary: Electricity Maps / CO2Signal (covers India)
# Set CO2SIGNAL_API_KEY env var to enable. Free tier available.
CO2SIGNAL_API_KEY = os.environ.get("CO2SIGNAL_API_KEY", "")

# India-specific zone code for Electricity Maps API
INDIA_ZONE_CODE   = os.environ.get("INDIA_CARBON_ZONE", "IN-SO")   # Southern India

# Fallback: UK National Grid API (public, no key needed)
UK_CARBON_API_URL = "https://api.carbonintensity.org.uk/intensity"

# Empirical average intensities (gCO2/kWh) from Ember 2024 Global Review
# Used when all live APIs fail
INDIA_STATE_FALLBACKS = {
    "Maharashtra": 659.0,
    "Telangana":   679.9,
    "Tamil Nadu":  493.2,
    "Delhi":       421.0,
    "default":     612.0,   # all-India average
}


class CarbonIntensityClient:
    """
    Fetches live carbon intensity (gCO2/kWh).

    Priority order:
      1. Electricity Maps / CO2Signal (India zone — requires free API key)
      2. UK National Grid API        (UK value — fallback for normalisation only)
      3. Ember 2024 static value     (hardcoded last resort)
    """

    TIMEOUT_SECONDS  = 5
    MAX_RETRIES      = 2
    RETRY_DELAY_SECS = 1.0

    def __init__(self, state: str = "Maharashtra"):
        self.logger  = logger
        self.state   = state
        self.fallback_intensity = INDIA_STATE_FALLBACKS.get(
            state, INDIA_STATE_FALLBACKS["default"]
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def fetch_live_intensity(self) -> int:
        """
        Returns live carbon intensity as int (gCO2/kWh).
        Tries multiple sources with retry logic before falling back.
        """
        result = self.fetch_intensity_with_source()
        return result["intensity"]

    def fetch_intensity_with_source(self) -> dict:
        """
        Returns {"intensity": int, "source": str, "zone": str}.
        Useful for audit trails and PR comment attribution.
        """
        self.logger.info("Fetching live carbon intensity (state=%s) ...", self.state)

        # 1. Try Electricity Maps / CO2Signal if key is set
        if CO2SIGNAL_API_KEY:
            result = self._fetch_co2signal()
            if result is not None:
                return result

        # 2. Try UK National Grid (no auth, always available)
        result = self._fetch_uk_grid()
        if result is not None:
            # UK value is used as a normalisation proxy only — log the caveat
            self.logger.warning(
                "Using UK grid intensity as proxy (%d gCO2/kWh). "
                "Set CO2SIGNAL_API_KEY for India-specific data.",
                result["intensity"],
            )
            return result

        # 3. Static Ember 2024 fallback
        self.logger.warning(
            "All live sources failed. Using Ember 2024 static value "
            "for %s: %d gCO2/kWh",
            self.state, self.fallback_intensity,
        )
        return {
            "intensity": int(self.fallback_intensity),
            "source":    "Ember 2024 static (fallback)",
            "zone":      self.state,
        }

    # ── Private fetchers ──────────────────────────────────────────────────────

    def _fetch_co2signal(self) -> Optional[dict]:
        """Electricity Maps / CO2Signal API — covers India zones."""
        url = (
            f"https://api.co2signal.com/v1/latest"
            f"?countryCode={INDIA_ZONE_CODE}"
        )
        headers = {
            "auth-token": CO2SIGNAL_API_KEY,
            "Accept":     "application/json",
        }
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=self.TIMEOUT_SECONDS) as resp:
                    if resp.status != 200:
                        raise ValueError(f"HTTP {resp.status}")
                    data = json.loads(resp.read().decode("utf-8"))
                    intensity = data.get("data", {}).get("carbonIntensity")
                    if intensity is not None:
                        self.logger.info(
                            "CO2Signal: %d gCO2/kWh (zone=%s)",
                            int(intensity), INDIA_ZONE_CODE
                        )
                        return {
                            "intensity": int(intensity),
                            "source":    "Electricity Maps / CO2Signal",
                            "zone":      INDIA_ZONE_CODE,
                        }
            except Exception as exc:
                self.logger.warning(
                    "CO2Signal attempt %d/%d failed: %s",
                    attempt, self.MAX_RETRIES, repr(exc)
                )
                if attempt < self.MAX_RETRIES:
                    time.sleep(self.RETRY_DELAY_SECS * attempt)
        return None

    def _fetch_uk_grid(self) -> Optional[dict]:
        """UK National Grid ESO API — public, no auth needed."""
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                req = urllib.request.Request(
                    UK_CARBON_API_URL,
                    headers={"Accept": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=self.TIMEOUT_SECONDS) as resp:
                    if resp.status != 200:
                        raise ValueError(f"HTTP {resp.status}")
                    payload = json.loads(resp.read().decode("utf-8"))
                    actual  = payload["data"][0]["intensity"]["actual"]
                    # FIX: guard against None before int cast
                    if actual is not None:
                        self.logger.info(
                            "UK Grid: %d gCO2/kWh (used as proxy only)", int(actual)
                        )
                        return {
                            "intensity": int(actual),
                            "source":    "UK National Grid ESO (proxy)",
                            "zone":      "GB",
                        }
                    self.logger.warning("UK Grid API returned null intensity.")
            except Exception as exc:
                self.logger.warning(
                    "UK Grid attempt %d/%d failed: %s",
                    attempt, self.MAX_RETRIES, repr(exc)
                )
                if attempt < self.MAX_RETRIES:
                    time.sleep(self.RETRY_DELAY_SECS * attempt)
        return None
