from openai import OpenAI
from src.config.settings import settings

class LLMAgent:
    def __init__(self):
        # If API key is not available (like in CI)
        if not settings.OPENAI_API_KEY:
            self.client = None
        else:
            self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def decide(self, similarity, carbon):
        # If no API client → fallback decision
        if not self.client:
            return "RUN_PARTIAL_TESTS"

        prompt = f"""
        Code similarity: {similarity}
        Carbon intensity: {carbon}

        Output ONLY:
        RUN_ALL_TESTS / RUN_PARTIAL_TESTS / SKIP_TESTS
        """

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content.strip()