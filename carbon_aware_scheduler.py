"""
carbon_aware_scheduler.py
==========================
Green-Ops CI/CD Framework — Carbon-Aware Test Scheduler

Takes the XGBoost pruning decision (run/prune lists + Pf scores) and decides:
  1. WHICH tests to run              (already decided by XGBoost gatekeeper)
  2. WHERE to run them               (which datacenter — based on carbon intensity)
  3. WHEN to run heavy tests         (defer if grid is dirty, run now if clean)
  4. HOW to batch them               (group by operation cost for efficient scheduling)

Scheduling logic:
  - Estimate total CPU operations for the full test set from AST metadata
  - Query SQLite (india_carbon_pipeline.py DB) for current intensity per DC
  - Default provider: AWS  (overridable via GREENOPS_PROVIDER env var)
  - Default selection: cleanest available datacenter across all providers
  - If operations > HEAVY_THRESHOLD and carbon_score > 0.6 → defer to off-peak
  - Emit a schedule JSON consumed by github_actions_runner.py

Datacenter routing:
  AWS   → ap-south-1 (Mumbai/Maharashtra)   or  ap-south-2 (Hyderabad/Telangana)
  Azure → centralindia (Pune/Maharashtra)   or  southindia (Chennai/Tamil Nadu)
  GCP   → asia-south1  (Mumbai/Maharashtra) or  asia-south2 (Delhi)

Carbon intensity values come from the Ember 2024 dataset loaded into SQLite
by india_carbon_pipeline.py.

Usage:
    from carbon_aware_scheduler import CarbonAwareScheduler
    scheduler = CarbonAwareScheduler(db_path="greenops.db")
    schedule  = scheduler.schedule(pruning_decision, test_operation_counts)
"""

import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

log = logging.getLogger("greenops.scheduler")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_PROVIDER     = os.environ.get("GREENOPS_PROVIDER", "aws").lower()
GREENOPS_DB          = os.environ.get("GREENOPS_DB", "greenops.db")
OUTPUT_DIR           = Path(os.environ.get("GREENOPS_OUTPUT", "./greenops_output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Operation thresholds for scheduling tiers
LIGHT_OPS_THRESHOLD  =   5_000   # < 5K ops  → light test, always schedule now
MEDIUM_OPS_THRESHOLD =  50_000   # < 50K ops → medium test
HEAVY_OPS_THRESHOLD  = 200_000   # > 200K ops → heavy test, consider deferral

# Carbon score above which we defer heavy tests to off-peak
DEFER_CARBON_SCORE   = 0.65      # normalized intensity / 900 > 0.65 → defer heavy

# Ember 2024 fallback intensities (gCO2/kWh) if DB not available
# Source: Ember Global Electricity Review 2024, CC BY 4.0
FALLBACK_INTENSITIES = {
    "aws":   {"zone": "ap-south-1",   "state": "Maharashtra", "intensity": 659.0},
    "azure": {"zone": "centralindia", "state": "Maharashtra", "intensity": 659.0},
    "gcp":   {"zone": "asia-south1",  "state": "Maharashtra", "intensity": 659.0},
}

# All Indian datacenter zones with their states
ALL_DC_ZONES = {
    "aws": [
        {"zone": "ap-south-1",  "city": "Mumbai",    "state": "Maharashtra"},
        {"zone": "ap-south-2",  "city": "Hyderabad", "state": "Telangana"},
    ],
    "azure": [
        {"zone": "centralindia","city": "Pune",      "state": "Maharashtra"},
        {"zone": "westindia",   "city": "Mumbai",    "state": "Maharashtra"},
        {"zone": "southindia",  "city": "Chennai",   "state": "Tamil Nadu"},
    ],
    "gcp": [
        {"zone": "asia-south1", "city": "Mumbai",    "state": "Maharashtra"},
        {"zone": "asia-south2", "city": "Delhi",     "state": "Delhi"},
    ],
}

# Operation cost weights (same as carbon_estimator.py for consistency)
OPERATION_COSTS = {
    "function_call": 10, "loop_iteration": 5, "comprehension": 8,
    "conditional": 2, "arithmetic": 1, "assignment": 1,
    "import": 50, "class_definition": 20, "exception_handler": 15,
    "context_manager": 8, "yield": 12, "lambda": 5,
    "boolean_op": 2, "subscript": 3, "attribute_access": 4, "augmented_assign": 2,
}

CPU_GHZ       = 3.0
CPU_TDP_WATTS = 15.0
JOULES_TO_KWH = 1 / 3_600_000


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DatacenterOption:
    provider:        str
    zone:            str
    city:            str
    state:           str
    intensity:       float       # gCO2/kWh from Ember dataset via SQLite
    carbon_score:    float       # normalized 0–1 (intensity / 900)
    is_provider_avg: bool = False


@dataclass
class TestScheduleEntry:
    test_name:       str
    pf_score:        float
    total_ops:       int
    tier:            str         # "light" | "medium" | "heavy"
    carbon_gco2:     float       # estimated carbon for this test
    assigned_dc:     str         # zone name e.g. "ap-south-1"
    assigned_city:   str
    assigned_state:  str
    provider:        str
    schedule_now:    bool        # True = run immediately, False = defer to off-peak
    defer_reason:    str = ""    # why it was deferred


@dataclass
class ScheduleResult:
    provider:               str
    selected_zone:          str
    selected_city:          str
    selected_state:         str
    carbon_intensity:       float
    carbon_score:           float
    total_tests_to_run:     int
    total_ops_estimated:    int
    total_carbon_gco2:      float
    schedule_now:           list = field(default_factory=list)
    schedule_deferred:      list = field(default_factory=list)
    historic_failure_tests: list = field(default_factory=list)
    all_dc_options:         list = field(default_factory=list)
    recommendation:         str  = ""
    data_source:            str  = "Ember 2024 via SQLite"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — DATACENTER INTENSITY LOADER
# ─────────────────────────────────────────────────────────────────────────────

class DatacenterIntensityLoader:
    """
    Reads carbon intensity for all datacenter zones from SQLite
    (populated by india_carbon_pipeline.py).
    Falls back to Ember 2024 hardcoded values if DB is missing.
    """

    def __init__(self, db_path: str = GREENOPS_DB):
        self.db_path = db_path
        self._conn   = None

    def _get_conn(self) -> Optional[sqlite3.Connection]:
        if self._conn is None:
            if not Path(self.db_path).exists():
                log.warning("DB not found at %s — using Ember 2024 fallback values", self.db_path)
                return None
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def load_all_options(self, year: int = 2024) -> list:
        """
        Load carbon intensity for every DC zone across all providers.
        Returns list of DatacenterOption sorted by intensity (cleanest first).
        """
        options = []
        conn    = self._get_conn()

        for provider, zones in ALL_DC_ZONES.items():
            for zone_info in zones:
                intensity = self._get_intensity(conn, zone_info["state"], year)
                options.append(DatacenterOption(
                    provider     = provider,
                    zone         = zone_info["zone"],
                    city         = zone_info["city"],
                    state        = zone_info["state"],
                    intensity    = intensity,
                    carbon_score = round(min(intensity / 900.0, 1.0), 4),
                ))

        options.sort(key=lambda x: x.intensity)
        log.info("Loaded %d DC zone options (cleanest: %s at %.0f gCO2/kWh)",
                 len(options), options[0].zone if options else "?",
                 options[0].intensity if options else 0)
        return options

    def load_provider_options(self, provider: str, year: int = 2024) -> list:
        """Load options for a specific provider only."""
        all_opts = self.load_all_options(year)
        return [o for o in all_opts if o.provider == provider.lower()]

    def _get_intensity(self, conn, state: str, year: int) -> float:
        """Get intensity for a state from DB, or use fallback."""
        if conn is None:
            # Fallback: use Maharashtra value (most DCs are here)
            fallback_map = {
                "Maharashtra": 659.0, "Telangana": 679.9,
                "Tamil Nadu":  493.2, "Delhi":     421.0,
            }
            return fallback_map.get(state, 659.0)

        try:
            row = conn.execute("""
                SELECT co2_intensity_gco2_kwh FROM state_carbon_intensity
                WHERE state = ? AND year = ?
            """, (state, year)).fetchone()
            if row:
                return float(row["co2_intensity_gco2_kwh"])
        except sqlite3.OperationalError:
            pass

        # Fallback if table not present
        fallback_map = {
            "Maharashtra": 659.0, "Telangana": 679.9,
            "Tamil Nadu":  493.2, "Delhi":     421.0,
        }
        return fallback_map.get(state, 659.0)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — OPERATION ESTIMATOR
# ─────────────────────────────────────────────────────────────────────────────

class OperationEstimator:
    """
    Estimates total CPU operations for a set of tests.
    Uses either pre-computed operation counts from AST metadata
    or falls back to a heuristic based on test name / historical duration.
    """

    def estimate_from_ast(self, test_op_counts: dict) -> dict:
        """
        test_op_counts: {test_name: {op_type: count, ...}}
        Returns: {test_name: total_weighted_ops}
        """
        result = {}
        for test_name, op_profile in test_op_counts.items():
            total = sum(
                op_profile.get(op, 0) * weight
                for op, weight in OPERATION_COSTS.items()
            )
            # Add raw counts not in OPERATION_COSTS directly
            total += op_profile.get("total_ops", 0)
            result[test_name] = max(total, 1)
        return result

    def estimate_from_duration(self, test_stats: dict) -> dict:
        """
        Fallback: estimate ops from historical test duration.
        test_stats: {test_name: {"test_duration": float_seconds}}
        Formula: ops ≈ duration_seconds × CPU_GHz × 1e9 × 0.001
                  (0.1% efficiency — most time is I/O, not pure compute)
        """
        result = {}
        for test_name, stats in test_stats.items():
            duration = stats.get("test_duration", 1.0) or 1.0
            # Conservative: assume 0.1% of cycles are actual compute ops
            ops = int(duration * CPU_GHZ * 1e9 * 0.001)
            result[test_name] = max(ops, 100)
        return result

    def estimate_carbon(self, total_ops: int, intensity: float) -> float:
        """
        Convert operation count → carbon (gCO2).
        ops → clock_cycles → joules → kWh → gCO2
        """
        joules_per_cycle = CPU_TDP_WATTS / (CPU_GHZ * 1e9)
        energy_joules    = total_ops * joules_per_cycle
        energy_kwh       = energy_joules * JOULES_TO_KWH
        return round(energy_kwh * intensity, 8)

    def classify_tier(self, total_ops: int) -> str:
        if total_ops < LIGHT_OPS_THRESHOLD:
            return "light"
        elif total_ops < MEDIUM_OPS_THRESHOLD:
            return "medium"
        elif total_ops < HEAVY_OPS_THRESHOLD:
            return "heavy"
        else:
            return "very_heavy"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — DATACENTER SELECTOR
# ─────────────────────────────────────────────────────────────────────────────

class DatacenterSelector:
    """
    Selects the best datacenter for running a test set.

    Selection rules (in priority order):
      1. If provider explicitly set → use that provider's cleanest zone
      2. Default provider = AWS     → use AWS's cleanest zone
      3. "cleanest" mode            → pick the absolute cleanest across all providers
      4. For heavy tests on dirty grid → recommend deferral
    """

    def select(
        self,
        all_options:      list,          # DatacenterOption list sorted by intensity
        provider:         str = "aws",   # "aws" | "azure" | "gcp" | "cleanest"
        force_zone:       Optional[str] = None,  # explicit zone override
    ) -> DatacenterOption:
        """
        Returns the selected DatacenterOption.
        """
        if force_zone:
            match = next((o for o in all_options if o.zone == force_zone), None)
            if match:
                log.info("Zone override: using %s (%s)", force_zone, match.city)
                return match
            log.warning("Zone %s not found — falling back to provider selection", force_zone)

        if provider == "cleanest":
            # Absolute cleanest across all providers
            best = min(all_options, key=lambda o: o.intensity)
            log.info("Cleanest DC selected: %s/%s (%.0f gCO2/kWh)",
                     best.provider.upper(), best.zone, best.intensity)
            return best

        # Provider-specific: pick cleanest zone within that provider
        provider_opts = [o for o in all_options if o.provider == provider.lower()]
        if not provider_opts:
            log.warning("No options for provider %s — using overall cleanest", provider)
            return min(all_options, key=lambda o: o.intensity)

        best = min(provider_opts, key=lambda o: o.intensity)
        log.info("Selected %s/%s — %s (%.0f gCO2/kWh, score=%.3f)",
                 provider.upper(), best.zone, best.city,
                 best.intensity, best.carbon_score)
        return best

    def should_defer(
        self,
        selected_dc: DatacenterOption,
        total_ops:   int,
        tier:        str,
    ) -> tuple:
        """
        Decide if a heavy test should be deferred to off-peak hours.

        Returns (should_defer: bool, reason: str)

        Deferral logic:
          - Light/medium tests → never defer (fast, low carbon cost)
          - Heavy/very_heavy tests on dirty grid (carbon_score > 0.65) → defer
          - Very heavy tests on any grid with score > 0.5 → consider defer
        """
        if tier in ("light", "medium"):
            return False, ""

        score = selected_dc.carbon_score

        if tier == "very_heavy" and score > 0.5:
            return True, (
                f"Very heavy test ({total_ops:,} ops) on {selected_dc.city} "
                f"grid ({selected_dc.intensity:.0f} gCO2/kWh, score={score:.2f}). "
                f"Defer to off-peak for ~{score*100:.0f}% carbon savings."
            )

        if tier == "heavy" and score > DEFER_CARBON_SCORE:
            return True, (
                f"Heavy test ({total_ops:,} ops) on high-carbon grid "
                f"({selected_dc.intensity:.0f} gCO2/kWh, score={score:.2f}). "
                f"Deferred — schedule during renewable-energy peak."
            )

        return False, ""


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SCHEDULER CLASS
# ─────────────────────────────────────────────────────────────────────────────

class CarbonAwareScheduler:
    """
    Main entry point for the carbon-aware test scheduler.

    Consumes the output of XGBoostGatekeeper.decide() and produces a
    full schedule: which tests run now, which are deferred, and where.

    Default behaviour:
      - Provider: AWS (override via GREENOPS_PROVIDER or constructor)
      - Zone: cleanest available zone within the provider
      - Heavy tests on dirty grid: deferred
      - Historic failure tests: always scheduled now, regardless of carbon
    """

    def __init__(
        self,
        db_path:  str = GREENOPS_DB,
        provider: str = DEFAULT_PROVIDER,
        zone:     Optional[str] = None,   # explicit zone e.g. "ap-south-1"
        year:     int = 2024,
    ):
        self.provider   = provider.lower()
        self.zone       = zone
        self.year       = year
        self.loader     = DatacenterIntensityLoader(db_path)
        self.estimator  = OperationEstimator()
        self.selector   = DatacenterSelector()
        self._all_opts  = None   # lazy loaded

    def _get_all_options(self) -> list:
        if self._all_opts is None:
            self._all_opts = self.loader.load_all_options(self.year)
        return self._all_opts

    def schedule(
        self,
        pruning_decision:    dict,   # output of XGBoostGatekeeper.decide()
        test_op_counts:      Optional[dict] = None,  # {test_name: {op_type: count}}
        test_duration_stats: Optional[dict] = None,  # {test_name: {test_duration: s}}
    ) -> dict:
        """
        Main scheduling method.

        Args:
            pruning_decision:    dict from XGBoostGatekeeper — has 'run', 'prune',
                                 'pf_scores', 'reasoning', 'historic_failure_tests'
            test_op_counts:      Optional AST-derived operation counts per test
            test_duration_stats: Optional historical duration stats per test

        Returns:
            Full schedule dict with DC assignment, carbon estimates, run/defer split
        """
        tests_to_run       = pruning_decision.get("run", [])
        pf_scores          = pruning_decision.get("pf_scores", {})
        historic_failures  = pruning_decision.get(
            "historic_failure_tests",
            pruning_decision.get("reasoning", {}).get("historic_failures", [])
        )

        if not tests_to_run and not historic_failures:
            log.info("No tests to schedule — all pruned by XGBoost.")
            return self._empty_schedule()

        log.info("=" * 60)
        log.info("Carbon-Aware Scheduler starting")
        log.info("Tests to schedule : %d", len(tests_to_run))
        log.info("Historic failures : %d (always run)", len(historic_failures))
        log.info("Provider          : %s", self.provider.upper())
        log.info("=" * 60)

        # ── Step 1: Load all DC options and select best ───────────────────────
        all_options     = self._get_all_options()
        selected_dc     = self.selector.select(all_options, self.provider, self.zone)

        # ── Step 2: Estimate operation counts per test ────────────────────────
        all_test_names = list(set(tests_to_run + [
            t["test_name"] if isinstance(t, dict) else t
            for t in historic_failures
        ]))

        if test_op_counts:
            op_estimates = self.estimator.estimate_from_ast(test_op_counts)
        elif test_duration_stats:
            op_estimates = self.estimator.estimate_from_duration(test_duration_stats)
        else:
            # Heuristic fallback: assume medium-sized test
            op_estimates = {t: 10_000 for t in all_test_names}

        # ── Step 3: Build schedule entries for XGBoost-selected tests ─────────
        schedule_now      = []
        schedule_deferred = []
        total_ops         = 0
        total_carbon      = 0.0

        for test_name in tests_to_run:
            ops      = op_estimates.get(test_name, 10_000)
            tier     = self.estimator.classify_tier(ops)
            carbon   = self.estimator.estimate_carbon(ops, selected_dc.intensity)
            pf       = pf_scores.get(test_name, 0.0)
            defer, reason = self.selector.should_defer(selected_dc, ops, tier)

            entry = asdict(TestScheduleEntry(
                test_name      = test_name,
                pf_score       = round(pf, 4),
                total_ops      = ops,
                tier           = tier,
                carbon_gco2    = carbon,
                assigned_dc    = selected_dc.zone,
                assigned_city  = selected_dc.city,
                assigned_state = selected_dc.state,
                provider       = selected_dc.provider,
                schedule_now   = not defer,
                defer_reason   = reason,
            ))

            total_ops    += ops
            total_carbon += carbon

            if defer:
                schedule_deferred.append(entry)
            else:
                schedule_now.append(entry)

        # ── Step 4: Historic failure tests — ALWAYS schedule now ──────────────
        historic_entries = []
        for t in historic_failures:
            test_name = t["test_name"] if isinstance(t, dict) else t
            ops       = op_estimates.get(test_name, 10_000)
            carbon    = self.estimator.estimate_carbon(ops, selected_dc.intensity)
            tier      = self.estimator.classify_tier(ops)
            total_ops    += ops
            total_carbon += carbon

            historic_entries.append(asdict(TestScheduleEntry(
                test_name      = test_name,
                pf_score       = t.get("failure_rate", 1.0) if isinstance(t, dict) else 1.0,
                total_ops      = ops,
                tier           = tier,
                carbon_gco2    = carbon,
                assigned_dc    = selected_dc.zone,
                assigned_city  = selected_dc.city,
                assigned_state = selected_dc.state,
                provider       = selected_dc.provider,
                schedule_now   = True,   # ALWAYS run — no deferral for historic failures
                defer_reason   = "",
            )))

        # ── Step 5: Build recommendation string ───────────────────────────────
        alt_cleanest = min(all_options, key=lambda o: o.intensity)
        recommendation = self._build_recommendation(
            selected_dc, alt_cleanest, total_ops,
            len(schedule_deferred), total_carbon
        )

        # ── Step 6: Assemble full result ──────────────────────────────────────
        result = asdict(ScheduleResult(
            provider               = selected_dc.provider,
            selected_zone          = selected_dc.zone,
            selected_city          = selected_dc.city,
            selected_state         = selected_dc.state,
            carbon_intensity       = selected_dc.intensity,
            carbon_score           = selected_dc.carbon_score,
            total_tests_to_run     = len(tests_to_run),
            total_ops_estimated    = total_ops,
            total_carbon_gco2      = round(total_carbon, 8),
            schedule_now           = schedule_now,
            schedule_deferred      = schedule_deferred,
            historic_failure_tests = historic_entries,
            all_dc_options         = [asdict(o) for o in all_options],
            recommendation         = recommendation,
        ))

        # Save schedule artifact
        schedule_path = OUTPUT_DIR / "test_schedule.json"
        with open(schedule_path, "w") as f:
            json.dump(result, f, indent=2)
        log.info("Schedule saved → %s", schedule_path)

        self._print_schedule_summary(result, all_options)
        return result

    def _build_recommendation(
        self,
        selected: DatacenterOption,
        cleanest: DatacenterOption,
        total_ops: int,
        n_deferred: int,
        total_carbon: float,
    ) -> str:
        parts = [
            f"Scheduled on {selected.provider.upper()}/{selected.zone} "
            f"({selected.city}, {selected.state}) — "
            f"{selected.intensity:.0f} gCO2/kWh."
        ]
        if cleanest.zone != selected.zone:
            savings = round(
                (selected.intensity - cleanest.intensity) / selected.intensity * 100, 1
            )
            parts.append(
                f"Cleanest alternative: {cleanest.provider.upper()}/{cleanest.zone} "
                f"({cleanest.city}) at {cleanest.intensity:.0f} gCO2/kWh "
                f"({savings}% cleaner)."
            )
        if n_deferred > 0:
            parts.append(
                f"{n_deferred} heavy test(s) deferred to off-peak for carbon savings."
            )
        parts.append(
            f"Estimated total carbon: {total_carbon:.6f} gCO2 "
            f"for {total_ops:,} operations."
        )
        return " ".join(parts)

    def _empty_schedule(self) -> dict:
        return {
            "provider": self.provider, "selected_zone": "", "selected_city": "",
            "total_tests_to_run": 0, "total_ops_estimated": 0,
            "total_carbon_gco2": 0.0, "schedule_now": [], "schedule_deferred": [],
            "historic_failure_tests": [], "recommendation": "No tests to schedule.",
        }

    @staticmethod
    def _print_schedule_summary(result: dict, all_options: list):
        print(f"\n{'='*65}")
        print("Carbon-Aware Test Schedule")
        print(f"{'='*65}")
        print(f"  Provider      : {result['provider'].upper()}")
        print(f"  Zone          : {result['selected_zone']}  ({result['selected_city']}, {result['selected_state']})")
        print(f"  Carbon        : {result['carbon_intensity']:.0f} gCO2/kWh  (score={result['carbon_score']:.3f})")
        print(f"  Total ops     : {result['total_ops_estimated']:,}")
        print(f"  Total carbon  : {result['total_carbon_gco2']:.6f} gCO2")
        print()
        print(f"  ✓ Run now     : {len(result['schedule_now'])} tests")
        print(f"  ⏸ Deferred    : {len(result['schedule_deferred'])} tests (heavy + dirty grid)")
        print(f"  ⚡ Historic   : {len(result['historic_failure_tests'])} tests (always run)")

        # Show all DC options ranked
        print(f"\n  All datacenter options (cleanest → dirtiest):")
        for o in all_options:
            zone = o.zone if hasattr(o, "zone") else o["zone"]
            prov = o.provider if hasattr(o, "provider") else o["provider"]
            city = o.city if hasattr(o, "city") else o["city"]
            intensity = o.intensity if hasattr(o, "intensity") else o["intensity"]
            marker = " ← SELECTED" if zone == result["selected_zone"] else ""
            print(f"    {prov.upper():<6} {zone:<16} {city:<12} "
                  f"{intensity:>7.0f} gCO2/kWh{marker}")

        if result.get("schedule_now"):
            print(f"\n  Tests scheduled NOW:")
            for e in sorted(result["schedule_now"],
                            key=lambda x: x["pf_score"], reverse=True)[:10]:
                print(f"    [{e['tier']:<10}] {e['test_name']:<40} "
                      f"Pf={e['pf_score']:.3f}  ops={e['total_ops']:>8,}")

        if result.get("schedule_deferred"):
            print(f"\n  Tests DEFERRED (off-peak):")
            for e in result["schedule_deferred"][:5]:
                print(f"    {e['test_name']}: {e['defer_reason'][:70]}")

        print(f"\n  Recommendation: {result['recommendation']}")
        print(f"{'='*65}\n")


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Green-Ops Carbon-Aware Scheduler")
    parser.add_argument("--decision",  default="./greenops_output/pruning_decision.json")
    parser.add_argument("--provider",  default=DEFAULT_PROVIDER,
                        choices=["aws", "azure", "gcp", "cleanest"])
    parser.add_argument("--zone",      default=None,
                        help="Explicit zone override e.g. ap-south-2")
    parser.add_argument("--db",        default=GREENOPS_DB)
    args = parser.parse_args()

    # Load pruning decision
    decision_path = Path(args.decision)
    if decision_path.exists():
        with open(decision_path) as f:
            pruning_decision = json.load(f)
    else:
        # Demo data
        pruning_decision = {
            "run":   ["PaymentServiceTest", "NotificationConsumerTest",
                      "AuditLoggerTest",    "PaymentIntegrationTest"],
            "prune": ["UserServiceTest", "StaticHelperTest"],
            "pf_scores": {
                "PaymentServiceTest":       0.82,
                "NotificationConsumerTest": 0.71,
                "AuditLoggerTest":          0.61,
                "PaymentIntegrationTest":   0.55,
            },
            "historic_failure_tests": [
                {"test_name": "PaymentE2ETest",  "failure_rate": 0.35, "total_runs": 40},
                {"test_name": "DBMigrationTest", "failure_rate": 0.18, "total_runs": 22},
            ],
        }

    # Demo operation counts (in real pipeline, from AST parser)
    test_op_counts = {
        "PaymentServiceTest":       {"function_call": 450, "loop_iteration": 200,
                                     "conditional": 300, "import": 15},
        "NotificationConsumerTest": {"function_call": 120, "loop_iteration": 80,
                                     "conditional": 90, "import": 8},
        "AuditLoggerTest":          {"function_call": 80,  "loop_iteration": 40,
                                     "conditional": 60,  "import": 5},
        "PaymentIntegrationTest":   {"function_call": 2000,"loop_iteration": 5000,
                                     "conditional": 3000, "import": 50,
                                     "comprehension": 400},   # heavy test
        "PaymentE2ETest":           {"function_call": 5000,"loop_iteration": 12000,
                                     "conditional": 8000, "import": 80},   # very heavy
        "DBMigrationTest":          {"function_call": 300, "loop_iteration": 800,
                                     "conditional": 200, "import": 20},
    }

    scheduler = CarbonAwareScheduler(
        db_path  = args.db,
        provider = args.provider,
        zone     = args.zone,
    )
    result = scheduler.schedule(pruning_decision, test_op_counts=test_op_counts)

    # Save readable summary
    summary_path = OUTPUT_DIR / "schedule_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "zone":           result["selected_zone"],
            "city":           result["selected_city"],
            "carbon_gco2_kwh": result["carbon_intensity"],
            "run_now":        [e["test_name"] for e in result["schedule_now"]],
            "deferred":       [e["test_name"] for e in result["schedule_deferred"]],
            "always_run":     [e["test_name"] for e in result["historic_failure_tests"]],
            "total_carbon":   result["total_carbon_gco2"],
            "recommendation": result["recommendation"],
        }, f, indent=2)
    print(f"Summary saved → {summary_path}")
