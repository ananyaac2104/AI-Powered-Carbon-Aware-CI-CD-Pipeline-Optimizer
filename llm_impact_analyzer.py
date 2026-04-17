"""
llm_impact_analyzer.py
======================
Green-Ops CI/CD Framework — LLM Impact Analyzer

Analyses the semantic impact of PR changes using an LLM, identifying:
  - Kafka topics affected
  - Shared DB tables touched
  - Downstream service dependencies
  - Contract changes (API, message schemas)
  - Summary for PR comment

Uses open-source models via Ollama (preferred, zero cost, local) or falls
back to Anthropic/OpenAI/Gemini API if configured. Never requires a paid API.

The LLMImpactAnalyzer is called by:
  - github_actions_runner.py (generates PR comment impact section)
  - pipeline_runner.py (Stage 3b: semantic enrichment)

USAGE:
    from llm_impact_analyzer import LLMImpactAnalyzer
    analyzer = LLMImpactAnalyzer()
    result = analyzer.analyze(
        changed_modules = [{"filepath": "src/auth.py", ...}],
        diff_text       = "...",
        dep_graph       = {...},
    )
    # result.summary, result.kafka_topics_affected, result.db_tables_affected
"""

import json
import logging
import os
import re
import time
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger("greenops.llm_impact_analyzer")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY    = os.environ.get("OPENAI_API_KEY", "")
GEMINI_API_KEY    = os.environ.get("GEMINI_API_KEY", "")
OLLAMA_BASE_URL   = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL      = os.environ.get("OLLAMA_MODEL", "codellama:7b")
GREENOPS_OUTPUT   = os.environ.get("GREENOPS_OUTPUT", "./greenops_output")
MAX_DIFF_CHARS    = 8000   # Truncate diff to stay within context windows
MAX_RETRIES       = 2
RETRY_DELAY       = 1.5

_SYSTEM_PROMPT = """You are a senior software architect analysing a pull request for
a carbon-aware CI/CD system. Given a list of changed source files and their
code diff, identify the full semantic impact of these changes.

Return ONLY valid JSON — no markdown, no prose, no code fences:
{
  "summary": "<2-3 sentence plain English description of what changed and why it matters for testing>",
  "kafka_topics_affected": ["topic1", "topic2"],
  "shared_db_tables_affected": ["table1", "table2"],
  "downstream_services_affected": ["service1"],
  "api_contracts_changed": true|false,
  "schema_migrations_present": true|false,
  "risk_level": "LOW"|"MEDIUM"|"HIGH"|"CRITICAL",
  "risk_reason": "<one sentence explaining risk level>",
  "recommended_test_tags": ["auth", "payment", "integration"],
  "safe_to_prune": ["test_unrelated_module", "test_docs_generation"]
}

Rules:
- Only include topics/tables/services you can identify from the diff.
- If no Kafka/DB/services are affected, return empty lists.
- safe_to_prune: only list tests you are highly confident are not affected.
- Never include the word "unclear" — if you can't determine something, omit it.
"""


# ─────────────────────────────────────────────────────────────────────────────
# RESULT DATA CLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ImpactAnalysis:
    """Structured result from LLM impact analysis."""
    summary:                    str = ""
    kafka_topics_affected:      List[str] = field(default_factory=list)
    shared_db_tables_affected:  List[str] = field(default_factory=list)
    downstream_services_affected: List[str] = field(default_factory=list)
    api_contracts_changed:      bool = False
    schema_migrations_present:  bool = False
    risk_level:                 str = "MEDIUM"
    risk_reason:                str = ""
    recommended_test_tags:      List[str] = field(default_factory=list)
    safe_to_prune:              List[str] = field(default_factory=list)
    provider_used:              str = ""
    analysis_time_ms:           float = 0.0
    fallback_used:              bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# LLM PROVIDERS
# ─────────────────────────────────────────────────────────────────────────────

class OllamaProvider:
    """Local Ollama inference — zero cost, no API key needed."""

    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = OLLAMA_MODEL):
        self.base_url = base_url.rstrip("/")
        self.model    = model

    def is_available(self) -> bool:
        try:
            req = urllib.request.Request(
                f"{self.base_url}/api/tags",
                headers={"Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                return resp.status == 200
        except Exception:
            return False

    def generate(self, prompt: str, system: str = "") -> Optional[str]:
        payload = json.dumps({
            "model":  self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 1024},
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data    = payload,
            headers = {"Content-Type": "application/json"},
            method  = "POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("response", "")
        except Exception as e:
            log.debug("Ollama error: %s", e)
            return None


class AnthropicProvider:
    def generate(self, prompt: str, system: str = "") -> Optional[str]:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            resp   = client.messages.create(
                model      = "claude-sonnet-4-20250514",
                max_tokens = 1024,
                system     = system,
                messages   = [{"role": "user", "content": prompt}],
            )
            return resp.content[0].text
        except Exception as e:
            log.warning("Anthropic error: %s", e)
            return None


class OpenAIProvider:
    def generate(self, prompt: str, system: str = "") -> Optional[str]:
        try:
            import openai
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            resp   = client.chat.completions.create(
                model    = "gpt-4o-mini",
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens  = 1024,
                temperature = 0.1,
            )
            return resp.choices[0].message.content
        except Exception as e:
            log.warning("OpenAI error: %s", e)
            return None


class GeminiProvider:
    def generate(self, prompt: str, system: str = "") -> Optional[str]:
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel("gemini-1.5-flash")
            resp  = model.generate_content(system + "\n\n" + prompt)
            return resp.text
        except Exception as e:
            log.warning("Gemini error: %s", e)
            return None


# ─────────────────────────────────────────────────────────────────────────────
# STATIC HEURISTIC ANALYZER (no LLM needed)
# ─────────────────────────────────────────────────────────────────────────────

def _static_heuristic_analysis(
    changed_modules: List[dict],
    diff_text:       str,
) -> ImpactAnalysis:
    """
    Deterministic static analysis fallback when no LLM is available.
    Scans diff text for keywords to infer impact.
    """
    diff_lower = diff_text.lower()
    code       = diff_text

    # Kafka detection
    kafka_patterns = [
        r"kafka[_\s.]*(topic|consumer|producer|stream)",
        r"KafkaProducer|KafkaConsumer|@KafkaListener",
        r"send\s*\(.*topic|subscribe\s*\(.*topic",
    ]
    kafka_topics = []
    for pat in kafka_patterns:
        matches = re.findall(pat, code, re.IGNORECASE)
        if matches:
            # Extract topic names near keywords
            topic_matches = re.findall(r"""[\"'`]([a-z][a-z0-9._\-]{2,40})[\"'`]""", code)
            kafka_topics.extend(topic_matches[:5])
            break

    # DB detection
    db_patterns  = [r"TABLE\s+(\w+)", r"FROM\s+(\w+)", r"INSERT\s+INTO\s+(\w+)", r"UPDATE\s+(\w+)\s+SET"]
    db_tables    = []
    for pat in db_patterns:
        matches = re.findall(pat, code, re.IGNORECASE)
        db_tables.extend(m for m in matches if len(m) > 2)

    # Risk assessment
    high_risk_patterns = [
        "migration", "schema", "alter table", "drop column", "rename column",
        "kafka", "message", "event", "contract", "interface", "api",
    ]
    high_risk_count = sum(1 for p in high_risk_patterns if p in diff_lower)
    risk_level = "HIGH" if high_risk_count >= 3 else "MEDIUM" if high_risk_count >= 1 else "LOW"

    # API contract changes
    api_contract = bool(re.search(r"@RestController|@RequestMapping|@GetMapping|@PostMapping|FastAPI|flask\.route|@app\.route", code))
    schema_mig   = bool(re.search(r"migration|alembic|flyway|liquibase|alter\s+table|add\s+column", code, re.IGNORECASE))

    # Test tags from module names
    changed_names = [m.get("filepath", m.get("file_path", "")) for m in changed_modules]
    tags = set()
    for name in changed_names:
        stem = Path(name).stem.lower()
        for tag in ["auth", "payment", "user", "order", "cart", "api", "db", "kafka", "email", "notification"]:
            if tag in stem:
                tags.add(tag)

    return ImpactAnalysis(
        summary                   = (
            f"PR modifies {len(changed_modules)} module(s): "
            f"{', '.join(Path(m.get('filepath', m.get('file_path', '?'))).name for m in changed_modules[:3])}. "
            f"Risk level: {risk_level}."
        ),
        kafka_topics_affected     = list(dict.fromkeys(kafka_topics))[:5],
        shared_db_tables_affected = list(dict.fromkeys(db_tables))[:8],
        api_contracts_changed     = api_contract,
        schema_migrations_present = schema_mig,
        risk_level                = risk_level,
        risk_reason               = f"{high_risk_count} high-risk patterns detected in diff",
        recommended_test_tags     = sorted(tags),
        safe_to_prune             = [],
        provider_used             = "static_heuristic",
        fallback_used             = True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ANALYZER CLASS
# ─────────────────────────────────────────────────────────────────────────────

class LLMImpactAnalyzer:
    """
    Analyses the semantic impact of PR changes using LLM or static heuristics.
    Provider priority: Ollama (local) → Anthropic → OpenAI → Gemini → static
    """

    def __init__(self):
        self._provider      = None
        self._provider_name = "none"
        self._select_provider()

    def _select_provider(self):
        # 1. Ollama (local, free, preferred)
        ollama = OllamaProvider()
        if ollama.is_available():
            self._provider      = ollama
            self._provider_name = f"ollama/{OLLAMA_MODEL}"
            log.info("LLM provider: %s (local)", self._provider_name)
            return

        # 2. API providers
        if ANTHROPIC_API_KEY:
            self._provider      = AnthropicProvider()
            self._provider_name = "anthropic/claude-sonnet"
        elif OPENAI_API_KEY:
            self._provider      = OpenAIProvider()
            self._provider_name = "openai/gpt-4o-mini"
        elif GEMINI_API_KEY:
            self._provider      = GeminiProvider()
            self._provider_name = "gemini/gemini-1.5-flash"
        else:
            log.info("No LLM provider available — static heuristic analysis only")

        if self._provider:
            log.info("LLM provider: %s", self._provider_name)

    def analyze(
        self,
        changed_modules: List[dict],
        diff_text:       str = "",
        dep_graph:       Optional[dict] = None,
        pr_number:       int = 0,
    ) -> ImpactAnalysis:
        """
        Run full impact analysis for a PR.

        Args:
            changed_modules: list of module dicts from preprocessing
            diff_text:       unified diff string (truncated internally)
            dep_graph:       optional pre-built dependency graph
            pr_number:       for caching/logging

        Returns ImpactAnalysis with all impact fields populated.
        """
        t0 = time.time()

        # Try cache first
        cached = self._load_cache(pr_number)
        if cached:
            log.info("Returning cached impact analysis for PR #%d", pr_number)
            return cached

        if not self._provider:
            result = _static_heuristic_analysis(changed_modules, diff_text)
        else:
            result = self._llm_analyze(changed_modules, diff_text, dep_graph)

        result.analysis_time_ms = round((time.time() - t0) * 1000, 1)
        self._save_cache(pr_number, result)

        # Also save as impact_analysis.json (expected by github_actions_runner)
        out_path = Path(GREENOPS_OUTPUT) / "impact_analysis.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        log.info("Impact analysis saved → %s", out_path)

        return result

    def _llm_analyze(
        self,
        changed_modules: List[dict],
        diff_text:       str,
        dep_graph:       Optional[dict],
    ) -> ImpactAnalysis:
        """Call LLM provider with retry logic."""
        prompt = self._build_prompt(changed_modules, diff_text, dep_graph)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                raw = self._provider.generate(prompt, _SYSTEM_PROMPT)
                if raw:
                    parsed = self._parse_response(raw)
                    if parsed:
                        parsed.provider_used = self._provider_name
                        return parsed
            except Exception as e:
                log.warning("LLM attempt %d/%d failed: %s", attempt, MAX_RETRIES, e)
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY * attempt)

        log.warning("LLM analysis failed after %d attempts — falling back to heuristic", MAX_RETRIES)
        result = _static_heuristic_analysis(changed_modules, diff_text)
        result.fallback_used  = True
        result.provider_used  = f"{self._provider_name}+static_fallback"
        return result

    @staticmethod
    def _build_prompt(
        changed_modules: List[dict],
        diff_text:       str,
        dep_graph:       Optional[dict],
    ) -> str:
        modules_summary = []
        for m in changed_modules[:10]:  # cap at 10
            fp   = m.get("filepath", m.get("file_path", "unknown"))
            lang = m.get("language", "unknown")
            fns  = m.get("functions", m.get("ast_result", {}).get("functions", []))
            fn_names = [f.get("name", f) if isinstance(f, dict) else str(f) for f in fns[:8]]
            modules_summary.append({
                "file":      fp,
                "language":  lang,
                "functions": fn_names,
            })

        # Truncate diff to keep prompt within context window
        truncated_diff = diff_text[:MAX_DIFF_CHARS]
        if len(diff_text) > MAX_DIFF_CHARS:
            truncated_diff += f"\n... [{len(diff_text) - MAX_DIFF_CHARS} chars truncated]"

        payload = {
            "changed_modules": modules_summary,
            "diff_excerpt":    truncated_diff,
        }
        if dep_graph:
            # Include only the relevant slice of the dep graph
            test_map   = dep_graph.get("test_map", {})
            rel_tests  = {}
            for m in changed_modules[:5]:
                fp = m.get("filepath", m.get("file_path", ""))
                if fp in test_map:
                    rel_tests[fp] = test_map[fp][:8]
            if rel_tests:
                payload["dependency_graph_excerpt"] = rel_tests

        return json.dumps(payload, indent=2)

    @staticmethod
    def _parse_response(raw: str) -> Optional[ImpactAnalysis]:
        """Parse and validate LLM JSON response."""
        clean = raw.strip()
        # Strip markdown fences
        clean = re.sub(r"^```(?:json)?\n?", "", clean).rstrip("`").strip()
        # Extract first JSON object
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if not match:
            log.debug("No JSON found in LLM response")
            return None
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError as e:
            log.debug("JSON parse error: %s", e)
            return None

        # Map to ImpactAnalysis with safe gets
        valid_risks = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        risk = str(data.get("risk_level", "MEDIUM")).upper()
        if risk not in valid_risks:
            risk = "MEDIUM"

        return ImpactAnalysis(
            summary                   = str(data.get("summary", ""))[:500],
            kafka_topics_affected     = [str(t) for t in data.get("kafka_topics_affected", [])[:10]],
            shared_db_tables_affected = [str(t) for t in data.get("shared_db_tables_affected", [])[:15]],
            downstream_services_affected = [str(s) for s in data.get("downstream_services_affected", [])[:10]],
            api_contracts_changed     = bool(data.get("api_contracts_changed", False)),
            schema_migrations_present = bool(data.get("schema_migrations_present", False)),
            risk_level                = risk,
            risk_reason               = str(data.get("risk_reason", ""))[:200],
            recommended_test_tags     = [str(t) for t in data.get("recommended_test_tags", [])[:10]],
            safe_to_prune             = [str(t) for t in data.get("safe_to_prune", [])[:20]],
        )

    def _cache_path(self, pr_number: int) -> Path:
        return Path(GREENOPS_OUTPUT) / f"impact_analysis_pr{pr_number}.json"

    def _save_cache(self, pr_number: int, result: ImpactAnalysis):
        if pr_number == 0:
            return
        try:
            with open(self._cache_path(pr_number), "w") as f:
                json.dump(result.to_dict(), f, indent=2)
        except Exception:
            pass

    def _load_cache(self, pr_number: int) -> Optional[ImpactAnalysis]:
        if pr_number == 0:
            return None
        path = self._cache_path(pr_number)
        if not path.exists():
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            return ImpactAnalysis(**{
                k: v for k, v in data.items()
                if k in ImpactAnalysis.__dataclass_fields__
            })
        except Exception:
            return None


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Green-Ops LLM Impact Analyzer")
    parser.add_argument("--diff",     help="Path to .diff file")
    parser.add_argument("--modules",  help="Path to preprocessing_artifacts JSON")
    parser.add_argument("--pr",       type=int, default=0, help="PR number")
    parser.add_argument("--output",   default=GREENOPS_OUTPUT)
    args = parser.parse_args()

    # Override output dir
    os.environ["GREENOPS_OUTPUT"] = args.output
    Path(args.output).mkdir(parents=True, exist_ok=True)

    diff_text = Path(args.diff).read_text() if args.diff and Path(args.diff).exists() else ""

    changed_modules = []
    if args.modules and Path(args.modules).exists():
        with open(args.modules) as f:
            data = json.load(f)
        changed_modules = data.get("modules", data if isinstance(data, list) else [])

    analyzer = LLMImpactAnalyzer()
    result   = analyzer.analyze(
        changed_modules = changed_modules,
        diff_text       = diff_text,
        pr_number       = args.pr,
    )

    print(f"\n{'='*60}")
    print("Impact Analysis Result")
    print(f"{'='*60}")
    print(f"Provider   : {result.provider_used}")
    print(f"Risk level : {result.risk_level}")
    print(f"Summary    : {result.summary}")
    print(f"Kafka      : {result.kafka_topics_affected}")
    print(f"DB tables  : {result.shared_db_tables_affected}")
    print(f"API changed: {result.api_contracts_changed}")
    print(f"Schema mig : {result.schema_migrations_present}")
    print(f"Tags       : {result.recommended_test_tags}")
    if result.safe_to_prune:
        print(f"Safe prune : {result.safe_to_prune[:5]}")
    print(f"Time       : {result.analysis_time_ms:.0f}ms")
    print(f"{'='*60}")
    print(f"\nSaved to {args.output}/impact_analysis.json")
