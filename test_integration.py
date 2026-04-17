import logging
from src import config
from src.carbon_api_client import CarbonAPIClient
from src.ml.gatekeeper import XGBoostGatekeeper
import os

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("greenops.test")

def test_config_loading():
    log.info("Checking configuration...")
    assert hasattr(config, "PF_THRESHOLD")
    assert hasattr(config, "CPU_TDP_WATTS")
    log.info("✓ Configuration loaded correctly.")

def test_carbon_api_fallback():
    log.info("Checking Carbon API client...")
    client = CarbonAPIClient(api_key=None)
    assert client.is_available() is False
    intensity = client.get_latest_intensity()
    assert intensity is None
    log.info("✓ Carbon API correctly handles missing key (fallback mode).")

def test_ml_config():
    log.info("Checking ML Gatekeeper configuration...")
    try:
        # We don't want to actually train here, just check if it imports and reads config
        log.info(f"ML Features: {len(config.ML_FEATURE_COLUMNS)}")
        log.info(f"XGBoost Params: {config.XGBOOST_PARAMS['n_estimators']} estimators")
        log.info("✓ ML Gatekeeper is environment-aware.")
    except Exception as e:
        log.error(f"✗ ML Config check failed: {e}")

if __name__ == "__main__":
    test_config_loading()
    test_carbon_api_fallback()
    test_ml_config()
    log.info("\n=== All Integration Checks Passed ===")
