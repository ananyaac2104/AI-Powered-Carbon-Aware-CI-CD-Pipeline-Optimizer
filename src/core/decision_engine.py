from src import config
from src.ml.gatekeeper import XGBoostGatekeeper
from src.core.carbon_resolver import get_intensity

class DecisionEngine:
    def __init__(self):
        self.gatekeeper = XGBoostGatekeeper()
        
    def decide(self, similarity, change_size, carbon_intensity=None):
        # 1. Fetch Carbon Intensity via Resolver (API -> DB -> Config)
        if carbon_intensity is None:
            carbon_intensity = get_intensity()
            
        # 2. Predict Failure Probability
        # Heuristic calculation for demo, now using tunable coefficients from config
        prob = (config.COEFFICIENT_SIM * (1 - similarity)) + \
               (config.COEFFICIENT_SIZE * (change_size / 100))
        
        # 3. Decision Logic using Config
        if prob >= config.PF_THRESHOLD:
            return {
                "decision": "RUN_ALL_TESTS",
                "reason": f"High risk detected (Pf={round(prob, 3)})",
                "probability": round(prob, 3),
                "carbon_intensity": carbon_intensity
            }

        # 4. AI Refinement (using Carbon Awareness)
        if carbon_intensity > config.DIRTY_GRID_THRESHOLD:
             return {
                "decision": "DEFER_TESTS",
                "reason": f"Dirty grid ({carbon_intensity} gCO2) exceeds threshold of {config.DIRTY_GRID_THRESHOLD}",
                "probability": round(prob, 3),
                "carbon_intensity": carbon_intensity
            }

        return {
            "decision": "SKIP_TESTS",
            "reason": "Optimized: Low risk and clean grid",
            "probability": round(prob, 3),
            "carbon_intensity": carbon_intensity
        }