import json
import logging
import urllib.request
from typing import Optional

logger = logging.getLogger("GreenOps.CarbonInference")

class CarbonIntensityClient:
    """Handles external network calls to the UK National Grid Carbon API."""
    
    API_URL = "https://api.carbonintensity.org.uk/intensity"
    TIMEOUT_SECONDS = 5
    FALLBACK_INTENSITY = 612
    
    def __init__(self):
        self.logger = logger
        
    def fetch_live_intensity(self) -> int:
        """
        Securely performs the HTTP request to fetch live gCO2/kWh intensity.
        Implements a fail-safe fallback mechanism on network timeouts or API drift.
        """
        self.logger.info("Initializing carbon intensity fetch protocols...")
        try:
            req = urllib.request.Request(self.API_URL, headers={'Accept': 'application/json'})
            with urllib.request.urlopen(req, timeout=self.TIMEOUT_SECONDS) as response:
                if response.status != 200:
                    raise ValueError(f"Unexpected HTTP Status: {response.status}")
                
                payload = json.loads(response.read().decode('utf-8'))
                actual_intensity = payload['data'][0]['intensity']['actual']
                
                if actual_intensity is not None:
                    self.logger.info(f"Successfully connected to Grid. Live Intensity: {actual_intensity} gCO2/kWh")
                    return int(actual_intensity)
                else:
                    self.logger.warning("API returned a null value for actual intensity.")
                    
        except Exception as e:
            self.logger.error(f"Carbon network request failed: {repr(e)}")
            
        self.logger.warning(f"Engaging fail-safe protocol -> Returning mock static value: {self.FALLBACK_INTENSITY}")
        return self.FALLBACK_INTENSITY
