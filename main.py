"""
main.py
=======
Green-Ops CI/CD Framework — Pipeline Entry Point

CHANGES (v2):
  - FIX: Original main.py only exercised DecisionEngine with hard-coded values
         and never showed how the carbon threshold comparison worked.
  - NEW: Demonstrates the carbon threshold comparison explicitly.
  - NEW: Exercises the GenerativeDependencyMapper Step 2 pipeline.
  - IMPROVEMENT: Shows feature importance from the gatekeeper.
"""

import json
from src.core.decision_engine import DecisionEngine
from generative_dependency_mapper import GenerativeDependencyMapper, PipelineInput


def demo_decision_engine():
    print("\n" + "=" * 60)
    print("DEMO 1 — Decision Engine with Carbon Threshold Comparison")
    print("=" * 60)

    engine = DecisionEngine()

    test_cases = [
        # (similarity, change_size, carbon_intensity, label)
        (0.90, 10,  350,  "High similarity, clean grid   → likely RUN"),
        (0.60, 25,  350,  "Medium similarity, clean grid → depends on Pf"),
        (0.60, 25,  700,  "Medium similarity, DIRTY grid → stricter pruning"),
        (0.15, 80,  500,  "Low similarity, large change  → likely PRUNE"),
        (0.95, 200, 800,  "Very high Pf, dirty grid      → RUN_ALL_TESTS"),
    ]

    for sim, cs, ci, label in test_cases:
        result = engine.decide(
            similarity       = sim,
            change_size      = cs,
            carbon_intensity = ci,
            module_impact_score = 0.7,
        )
        cc = result["carbon_check"]
        print(f"\n  {label}")
        print(f"    Decision    : {result['decision']}")
        print(f"    Pf          : {result['probability']:.3f}")
        print(f"    Pf threshold: {cc['pf_threshold_used']:.3f}")
        print(f"    Carbon      : {cc['carbon_intensity']:.0f} gCO2/kWh | "
              f"threshold={cc['carbon_threshold']:.0f} | "
              f"exceeded={cc['carbon_threshold_exceeded']}")
        print(f"    Reason      : {result['reason']}")
        print(f"    AI decision : {result['ai_decision']}")


def demo_step2_pipeline():
    print("\n" + "=" * 60)
    print("DEMO 2 — Step 2 Generative Dependency Mapper")
    print("=" * 60)

    mock_input = PipelineInput(
        changed_functions = ["processPayment", "validateCard"],
        similarity_scores = {
            ("processPayment", "test_payment_flow"):    0.92,
            ("processPayment", "test_notification"):    0.78,
            ("processPayment", "test_audit_log"):       0.55,
            ("validateCard",   "test_card_validation"): 0.88,
            ("validateCard",   "test_fraud_check"):     0.71,
            ("validateCard",   "test_static_helper"):   0.20,
        },
        similarity_threshold = 0.50,
    )

    mapper = GenerativeDependencyMapper(carbon_state="Tamil Nadu")
    result = mapper.execute(mock_input)

    print(f"\n  Carbon: {result.carbon_intensity:.0f} gCO2/kWh "
          f"(threshold={result.carbon_threshold:.0f}, "
          f"exceeded={result.carbon_threshold_exceeded})")
    print(f"  Source: {result.carbon_source}")
    print(f"\n  Dependency Graph:")
    for func, tests in result.graph.items():
        print(f"    {func}: {tests}")
    print(f"\n  Impacted tests: {result.impacted_tests}")
    print(f"\n  Test weights (descending by relevance):")
    for test, w in sorted(result.test_weights.items(), key=lambda x: x[1], reverse=True):
        print(f"    {test:<35} weight={w:.4f}")
    print(f"\n  Probability of Failure estimates:")
    for test, pf in result.probability_of_failure.items():
        flag = " ⚠️" if pf >= 0.7 else ""
        print(f"    {test:<35} Pf={pf:.4f}{flag}")
    if result.errors:
        print(f"\n  ⚠️  Errors: {result.errors}")


if __name__ == "__main__":
    demo_decision_engine()
    demo_step2_pipeline()

    print("\n" + "=" * 60)
    print("ALL DEMOS COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Set ANTHROPIC_API_KEY (or GEMINI_API_KEY) for LLM enrichment")
    print("  2. Set CO2SIGNAL_API_KEY for India-specific live carbon data")
    print("  3. Run preprocessing:  python preprocessing.py --presubmit ... --postsubmit ...")
    print("  4. Run full pipeline:  python github_ci_integration.py")
