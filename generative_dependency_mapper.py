"""
generative_dependency_mapper.py
===============================
Green-Ops CI/CD Framework — Generative Layer & Dependency Mapper (Step 2)

Main orchestrator facade that delegates to the micro-architecture layers:
- Carbon API Network
- Dynamic Graph Builder
- Generative LLM Check
"""

import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass, field

from carbon_inference_engine import CarbonIntensityClient
from dynamic_graph_builder import DynamicGraphBuilder
from llm_generative_agent import GenerativeGraphEnhancer

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("GreenOps.Orchestrator")

# ─────────────────────────────────────────────────────────────────────────────
# DATA CONTRACTS (DATACLASSES)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PipelineInput:
    """Immutable data contract encapsulating Akshaya's Step 1 output."""
    changed_functions: List[str]
    similarity_scores: Dict[Tuple[str, str], float]
    embeddings: Dict[str, List[float]] = field(default_factory=dict)
    similarity_threshold: float = 0.5


@dataclass
class PipelineOutput:
    """Strict data contract designed explicitly for Ananya's Step 3 ML layer."""
    graph: Dict[str, List[str]]
    impacted_tests: List[str]
    carbon_intensity: int


# ─────────────────────────────────────────────────────────────────────────────
# CORE ORCHESTRATOR FACADE
# ─────────────────────────────────────────────────────────────────────────────

class GenerativeDependencyMapper:
    """
    Main orchestrator for Step 2 of the Architecture.
    Facade pattern handing internal delegation to instantiated network and builder objects.
    """
    
    def __init__(self):
        self.logger = logger
        self.graph_builder = DynamicGraphBuilder()
        self.llm_enhancer = GenerativeGraphEnhancer()
        self.carbon_client = CarbonIntensityClient()

    def execute(self, payload: PipelineInput) -> PipelineOutput:
        """
        Executes the entire Step 2 architecture pipeline safely.
        """
        self.logger.info("========== STEP 2 PIPELINE COMMENCING ==========")
        
        # 1. Dependency Graph (Static Semantic Layer)
        graph, tests = self.graph_builder.build_heuristic_graph(
            changed_functions=payload.changed_functions,
            similarity_scores=payload.similarity_scores,
            similarity_threshold=payload.similarity_threshold
        )
        self.logger.info(f"Dependency mapped: {len(graph)} function branches, {len(tests)} affected tests.")
        
        # 2. Generative Component Edge Verification (Dynamic Layer)
        enriched_graph = self.llm_enhancer.verify_and_enrich_graph(graph)
        
        # 3. Operations Component (Carbon Integrations)
        actual_intensity = self.carbon_client.fetch_live_intensity()

        self.logger.info("\n ========== STEP 2 PIPELINE SUCCESSFUL ==========")
        
        return PipelineOutput(
            graph=enriched_graph,
            impacted_tests=tests,
            carbon_intensity=actual_intensity
        )


# ─────────────────────────────────────────────────────────────────────────────
# MANUAL EXECUTION AND VERIFICATION DEMO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mock_input = PipelineInput(
        changed_functions=["add", "multiply"],
        similarity_scores={
            ("add", "test_add"): 0.92,
            ("add", "test_db"): 0.12,
            ("multiply", "test_mul"): 0.89,
        },
        similarity_threshold=0.5
    )

    layer2_orchestrator = GenerativeDependencyMapper()
    layer2_result = layer2_orchestrator.execute(mock_input)

    print("\n" + "="*60)
    print("MULTI-FILE ARCHITECTURE OUTPUT")
    print("="*60)
    
    print("graph = {")
    for key, val in layer2_result.graph.items():
        print(f'    "{key}": {str(val).replace(chr(39), chr(34))},')
    print("}")
    
    print(f"\nimpacted_tests = {str(layer2_result.impacted_tests).replace(chr(39), chr(34))}")
    print(f"\ncarbon_intensity = {layer2_result.carbon_intensity}")
    print("="*60 + "\n")
