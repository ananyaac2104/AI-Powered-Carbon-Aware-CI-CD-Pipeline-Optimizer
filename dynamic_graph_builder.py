import logging
from typing import Dict, List, Tuple, Set

logger = logging.getLogger("GreenOps.GraphBuilder")

class DynamicGraphBuilder:
    """Constructs and validates the dependency graph connecting functions to tests."""
    
    def __init__(self):
        self.logger = logger

    def build_heuristic_graph(self, 
                              changed_functions: List[str], 
                              similarity_scores: Dict[Tuple[str, str], float],
                              similarity_threshold: float) -> Tuple[Dict[str, List[str]], List[str]]:
        """
        Builds the static base layer of the dependency graph using predefined semantic thresholds.
        """
        self.logger.info("Building core dependency graph utilizing semantic similarities...")
        graph: Dict[str, Set[str]] = {func: set() for func in changed_functions}
        impacted_tests: Set[str] = set()

        for (func, test), score in similarity_scores.items():
            if func in changed_functions and score >= similarity_threshold:
                graph[func].add(test)
                impacted_tests.add(test)

        final_graph = {k: sorted(list(v)) for k, v in graph.items()}
        return final_graph, sorted(list(impacted_tests))
