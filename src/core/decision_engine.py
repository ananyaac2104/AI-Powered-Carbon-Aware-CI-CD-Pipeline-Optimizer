from src.ml.gatekeeper import Gatekeeper
from src.ai.llm_agent import LLMAgent
from src.config.settings import settings

class DecisionEngine:
    def __init__(self):
        self.gatekeeper = Gatekeeper()
        self.llm = LLMAgent()

    def decide(self, similarity, change_size, carbon_intensity):
        prob = self.gatekeeper.predict_failure_prob(similarity, change_size)

        # Safety rule (industrial practice)
        if prob >= settings.FAILURE_THRESHOLD:
            return {
                "decision": "RUN_ALL_TESTS",
                "reason": "High failure probability",
                "probability": prob
            }

        # AI refinement
        ai_decision = self.llm.decide(similarity, carbon_intensity)

        return {
            "decision": ai_decision,
            "reason": "AI optimized",
            "probability": prob
        }