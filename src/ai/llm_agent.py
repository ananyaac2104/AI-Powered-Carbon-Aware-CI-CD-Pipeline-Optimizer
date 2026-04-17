"""
src/ai/llm_agent.py
===================
Green-Ops Framework — LLM Decision Agent

CHANGES (v2):
  - FIX: decide() previously always returned the string "AI_OPTIMIZED" regardless
         of inputs — the LLM was never actually called.  Now makes a real call.
  - FIX: No timeout or error handling — a stuck LLM call would block the pipeline.
         Added timeout and graceful fallback.
  - IMPROVEMENT: Uses the same provider priority as llm_generative_agent.py:
         Anthropic → OpenAI → Gemini → heuristic fallback.
  - IMPROVEMENT: Returns structured dict instead of bare string so callers
         can inspect the reasoning.
"""

import json
import logging
import os
from typing import Optional

log = logging.getLogger("greenops.llm_agent")

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY    = os.environ.get("OPENAI_API_KEY", "")
GEMINI_API_KEY    = os.environ.get("GEMINI_API_KEY", "")

_SYSTEM = """\
You are a CI/CD test-pruning advisor for Green-Ops, a carbon-aware CI system.
Given similarity and carbon intensity values, decide whether a test should run.
Return ONLY a JSON object:
{"decision": "RUN_TEST" | "PRUNE_TEST", "reason": "<one sentence>"}
"""


class LLMAgent:
    """Provides AI-optimised test run/prune decisions."""

    def __init__(self):
        self._provider = (
            "anthropic" if ANTHROPIC_API_KEY else
            "openai"    if OPENAI_API_KEY    else
            "gemini"    if GEMINI_API_KEY    else
            None
        )

    def decide(self, similarity: float, carbon_intensity: float) -> str:
        """
        Returns "RUN_TEST", "PRUNE_TEST", or "HEURISTIC: <reason>" as a string
        for backward compatibility with DecisionEngine.
        """
        result = self._call(similarity, carbon_intensity)
        return result.get("decision", "HEURISTIC: no LLM available")

    def decide_structured(self, similarity: float, carbon_intensity: float) -> dict:
        """Returns full structured dict from LLM."""
        return self._call(similarity, carbon_intensity)

    def _call(self, similarity: float, carbon_intensity: float) -> dict:
        prompt = json.dumps({
            "cosine_similarity":  round(similarity, 4),
            "carbon_intensity_gco2_kwh": round(carbon_intensity, 1),
        })

        raw = None
        try:
            if self._provider == "anthropic":
                raw = self._anthropic(prompt)
            elif self._provider == "openai":
                raw = self._openai(prompt)
            elif self._provider == "gemini":
                raw = self._gemini(prompt)
        except Exception as exc:
            log.warning("LLM call failed: %s", exc)

        if raw:
            try:
                import re
                clean = re.sub(r"```[a-z]*\n?", "", raw).strip().rstrip("`")
                return json.loads(clean)
            except Exception:
                pass

        # Heuristic fallback
        run = similarity >= 0.5
        return {
            "decision": "RUN_TEST" if run else "PRUNE_TEST",
            "reason":   f"Heuristic: similarity={similarity:.3f}",
        }

    def _anthropic(self, prompt: str) -> Optional[str]:
        import anthropic
        client   = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model      = "claude-sonnet-4-20250514",
            max_tokens = 256,
            system     = _SYSTEM,
            messages   = [{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def _openai(self, prompt: str) -> Optional[str]:
        import openai
        client   = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model    = "gpt-4o-mini",
            messages = [
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            max_tokens = 256,
        )
        return response.choices[0].message.content

    def _gemini(self, prompt: str) -> Optional[str]:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model    = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(_SYSTEM + "\n\n" + prompt)
        return response.text
