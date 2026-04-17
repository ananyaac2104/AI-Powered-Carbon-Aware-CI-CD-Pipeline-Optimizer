import logging
import requests
from typing import Optional, Dict, Any
from . import config

log = logging.getLogger("greenops.carbon_api")

class CarbonAPIClient:
    """
    Client for Electricity Maps API (v3).
    Provides real-time carbon intensity and forecasts for grid zones.
    """
    BASE_URL = "https://api.electricitymaps.com/v3"

    def __init__(self, api_key: Optional[str] = config.ELECTRICITY_MAPS_API_KEY):
        self.api_key = api_key
        self.headers = {"auth-token": self.api_key} if self.api_key else {}

    def is_available(self) -> bool:
        """Checks if the API key is configured."""
        return bool(self.api_key)

    def get_latest_intensity(self, zone: str = config.DEFAULT_ZONE) -> Optional[Dict[str, Any]]:
        """
        Fetch the last known carbon intensity for a specified zone.
        Returns: { 'zone', 'carbonIntensity', 'datetime', 'updatedAt' }
        """
        if not self.is_available():
            log.warning("Electricity Maps API key not set. Skipping API call.")
            return None

        url = f"{self.BASE_URL}/carbon-intensity/latest"
        params = {"zone": zone}
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            log.info(f"Fetched latest intensity for {zone}: {data.get('carbonIntensity')} gCO2eq/kWh")
            return data
        except Exception as e:
            log.error(f"Failed to fetch carbon intensity from Electricity Maps: {e}")
            return None

    def get_forecast(self, zone: str = config.DEFAULT_ZONE) -> Optional[Dict[str, Any]]:
        """
        Fetch the carbon intensity forecast for the next 24 hours.
        """
        if not self.is_available():
            return None

        url = f"{self.BASE_URL}/carbon-intensity/forecast"
        params = {"zone": zone}

        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            log.info(f"Fetched intensity forecast for {zone}")
            return data
        except Exception as e:
            log.error(f"Failed to fetch carbon forecast from Electricity Maps: {e}")
            return None
