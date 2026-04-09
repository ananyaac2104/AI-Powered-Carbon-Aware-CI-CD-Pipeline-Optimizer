import os

class Settings:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    FAILURE_THRESHOLD = 0.7

settings = Settings()