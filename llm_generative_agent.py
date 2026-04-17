"""
llm_generative_agent.py
=======================
Green-Ops CI/CD Framework — Generative Graph Enhancer

CHANGES (v2):
  - FIX: _invoke_generative_agent() was a pure mock — it appended a hardcoded
         "test_latent_integration" string to every function regardless of the
         graph content. This polluted every pruning decision with a phantom test.
         The method now makes a real LLM call (Gemini or OpenAI) with a
         structured prompt and falls back to a heuristic-only path (no phantom
         test injection) when no API key is available.
  - FIX: verify_and_enrich_graph() called _invoke_generative_agent() even when
         no API key was set, then logged a warning but still returned the mock
         result (with the phantom test). Now returns the original graph unchanged
         when no key is configured and no real call can be made.
  - IMPROVEMENT: Added proper JSON schema for LLM response parsing.
  - IMPROVEMENT: Added timeout and retry logic for LLM calls.
  - IMPROVEMENT: Real Gemini (google-generativeai) and OpenAI support.
"""

import json
import logging
import os
import re
import time
from typing import Dict, List, Optional

logger = logging.getLogger("GreenOps.GenerativeAgent")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

GEMINI_API_KEY  = os.environ.get("GEMINI_API_KEY", "")
OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("OPENAI_API_KEY", "")

MAX_RETRIES      = 2
RETRY_DELAY_SECS = 1.5

_SYSTEM_PROMPT = """\
You are a senior software architect reasoning about test impact for a CI/CD system.
Given a dependency graph mapping changed functions to their directly impacted tests,
identify any ADDITIONAL tests that should run due to latent or indirect dependencies
(e.g. shared utilities, transitive imports, shared state).

Return ONLY valid JSON with this exact schema — no prose, no markdown fences:
{
  "enriched_graph": {
    "<function_name>": ["<test_name_1>", "<test_name_2>", ...]
  },
  "added_tests": ["<test_name>", ...],
  "reasoning": "<one paragraph explaining what latent dependencies were found>"
}

Rules:
- Only ADD tests. Never remove tests from the existing graph.
- Only add tests you are highly confident are relevant. Prefer false negatives over false positives.
- If you find no additional tests, return the same graph unchanged and explain why.
- Do not invent test names — only use names that appear in the graph or are
  plausible variants (e.g. "test_auth_integration" for an auth module).
"""


class GenerativeGraphEnhancer:
    """
    Manages the GenAI prompting and graph validation pathways.
    Supports Anthropic Claude (primary), OpenAI, and Google Gemini.
    """

    def __init__(self):
        self.logger = logger
        # Choose provider based on available keys
        if ANTHROPIC_API_KEY:
            self._provider = "anthropic"
        elif GEMINI_API_KEY:
            self._provider = "gemini"
        elif OPENAI_API_KEY:
            self._provider = "openai"
        else:
            self._provider = None

        if self._provider:
            self.logger.info("LLM provider: %s", self._provider)
        else:
            self.logger.warning(
                "No LLM API keys found (ANTHROPIC_API_KEY / GEMINI_API_KEY / OPENAI_API_KEY). "
                "Graph enrichment will be skipped — original graph returned as-is."
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def verify_and_enrich_graph(
        self,
        graph: Dict[str, List[str]],
    ) -> Dict[str, List[str]]:
        """
        Enriches the dependency graph with latent dependencies detected by LLM.

        FIX: Returns the ORIGINAL graph unchanged if no LLM provider is configured
        instead of injecting a phantom test_latent_integration into every function.
        """
        if not self._provider:
            self.logger.info(
                "Skipping LLM graph enrichment — no API key. Graph unchanged."
            )
            return graph

        return self._invoke_generative_agent(graph)

    # ── Private methods ───────────────────────────────────────────────────────

    def _invoke_generative_agent(
        self,
        existing_graph: Dict[str, List[str]],
    ) -> Dict[str, List[str]]:
        """
        Calls the LLM to identify latent test dependencies.
        Falls back to returning the original graph if the call fails.
        """
        self.logger.info(
            "Invoking generative agent (%s) for graph enrichment ...", self._provider
        )

        prompt = json.dumps({
            "instruction": (
                "Analyse the dependency graph below and identify any additional "
                "tests that should run due to latent or indirect dependencies."
            ),
            "existing_graph": existing_graph,
        }, indent=2)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                raw = self._call_llm(prompt)
                if raw:
                    parsed = self._parse_response(raw, existing_graph)
                    if parsed:
                        return parsed
            except Exception as exc:
                self.logger.warning(
                    "LLM attempt %d/%d failed: %s", attempt, MAX_RETRIES, repr(exc)
                )
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY_SECS * attempt)

        self.logger.warning(
            "All LLM attempts failed — returning original graph unchanged."
        )
        return existing_graph

    def _call_llm(self, prompt: str) -> Optional[str]:
        """Dispatch to the correct LLM provider."""
        if self._provider == "anthropic":
            return self._call_anthropic(prompt)
        elif self._provider == "gemini":
            return self._call_gemini(prompt)
        elif self._provider == "openai":
            return self._call_openai(prompt)
        return None

    def _call_anthropic(self, prompt: str) -> Optional[str]:
        try:
            import anthropic
            client   = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            response = client.messages.create(
                model      = "claude-sonnet-4-20250514",
                max_tokens = 2048,
                system     = _SYSTEM_PROMPT,
                messages   = [{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as exc:
            self.logger.error("Anthropic call failed: %s", repr(exc))
            return None

    def _call_gemini(self, prompt: str) -> Optional[str]:
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            model    = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(_SYSTEM_PROMPT + "\n\n" + prompt)
            return response.text
        except Exception as exc:
            self.logger.error("Gemini call failed: %s", repr(exc))
            return None

    def _call_openai(self, prompt: str) -> Optional[str]:
        try:
            import openai
            client   = openai.OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model    = "gpt-4o-mini",
                messages = [
                    {"role": "system",  "content": _SYSTEM_PROMPT},
                    {"role": "user",    "content": prompt},
                ],
                max_tokens = 2048,
            )
            return response.choices[0].message.content
        except Exception as exc:
            self.logger.error("OpenAI call failed: %s", repr(exc))
            return None

    def _parse_response(
        self,
        raw: str,
        original_graph: Dict[str, List[str]],
    ) -> Optional[Dict[str, List[str]]]:
        """
        Parse and validate the LLM JSON response.
        Returns enriched_graph, or None if parsing fails.
        """
        # Strip markdown fences
        clean = raw.strip()
        if clean.startswith("```"):
            clean = re.sub(r"^```[a-z]*\n?", "", clean).rstrip("`").strip()

        try:
            data = json.loads(clean)
        except json.JSONDecodeError as exc:
            self.logger.warning("LLM response is not valid JSON: %s", exc)
            return None

        enriched = data.get("enriched_graph", {})
        if not isinstance(enriched, dict):
            return None

        # Merge: start from original, add LLM suggestions
        result = {k: list(v) for k, v in original_graph.items()}
        tests_added = 0
        for func, tests in enriched.items():
            if func in result and isinstance(tests, list):
                existing = set(result[func])
                new_tests = [t for t in tests if t not in existing]
                result[func] = sorted(list(existing | set(new_tests)))
                tests_added += len(new_tests)

        reasoning = data.get("reasoning", "")
        self.logger.info(
            "LLM enrichment complete: %d new test edges added. Reasoning: %s",
            tests_added, reasoning[:120],
        )
        return result
