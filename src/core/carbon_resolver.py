import sqlite3
import logging
from typing import Optional
from src import config
from src.carbon_api_client import CarbonAPIClient

log = logging.getLogger("greenops.carbon_resolver")

class CarbonResolver:
    """
    Tiered resolver for carbon intensity:
    1. Electricity Maps API (Real-time)
    2. Local greenops.db (Historical Ember India data)
    3. Last-resort fallback from config
    """

    def __init__(self):
        self.api_client = CarbonAPIClient()

    def get_current_intensity(self, zone: str = config.DEFAULT_ZONE) -> float:
        """Fetch carbon intensity with automated fallbacks."""
        
        # Priority 1: Real-time API
        try:
            api_data = self.api_client.get_latest_intensity(zone)
            if api_data and "carbonIntensity" in api_data:
                intensity = float(api_data["carbonIntensity"])
                log.info(f"Carbon intensity from API ({zone}): {intensity} gCO2/kWh")
                return intensity
        except Exception as e:
            log.warning(f"Electricity Maps API failed: {e}")

        # Priority 2: Local Database (Ember India data)
        try:
            db_intensity = self._fetch_from_db(zone)
            if db_intensity:
                log.info(f"Carbon intensity from DB fallback ({zone}): {db_intensity} gCO2/kWh")
                return db_intensity
        except Exception as e:
            log.warning(f"GreenOps DB lookup failed: {e}")

        # Priority 3: Config fallback (Last resort)
        log.error(f"All carbon sources failed for {zone}. Using hardcoded fallback: {config.DEFAULT_CARBON_FALLBACK}")
        return config.DEFAULT_CARBON_FALLBACK

    def _fetch_from_db(self, zone: str) -> Optional[float]:
        """Lookup historical intensity in SQLite greenops.db."""
        # Simple mapping for common zones if they are in India
        # Example: IN-MH -> Maharashtra
        zone_map = {
            "IN-SO": "South Region",
            "IN-MH": "Maharashtra",
            "IN-TG": "Telangana",
            "IN-TN": "Tamil Nadu",
            "IN-DL": "Delhi",
            "IN-KA": "Karnataka"
        }
        state_name = zone_map.get(zone, zone)

        try:
            conn = sqlite3.connect(config.GREENOPS_DB)
            cursor = conn.cursor()
            
            # Try to get the latest year's data for that state
            cursor.execute("""
                SELECT carbon_intensity_gco2_kwh 
                FROM state_carbon_intensity 
                WHERE state = ? 
                ORDER BY year DESC 
                LIMIT 1
            """, (state_name,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return float(row[0])
        except Exception:
            pass
        return None

# Global instance for easy access
resolver = CarbonResolver()

def get_intensity(zone: Optional[str] = None) -> float:
    """Helper function to get intensity from the global resolver instance."""
    return resolver.get_current_intensity(zone or config.DEFAULT_ZONE)
