"""
dynamic_graph_builder.py
========================
Green-Ops CI/CD Framework — Dynamic Dependency Graph Builder

CHANGES (v2):
  - FIX: build_heuristic_graph() used a bare dict of Sets but the `graph` type
         annotation said Dict[str, List[str]]; callers expected lists.
         Now consistently returns lists (sorted for determinism).
  - FIX: No deduplication of tests across functions — same test could appear
         multiple times in impacted_tests if matched by multiple functions.
         Now uses a set internally and converts at the end.
  - IMPROVEMENT: Added cycle detection (BFS) and exposes reachable_tests()
         for transitive test impact (depth > 1 in the call graph).
  - IMPROVEMENT: Added weighted_impact_scores() — computes a float impact
         weight per test based on sum of similarity scores from all matching
         functions, not just a binary include/exclude.
  - IMPROVEMENT: Graph serialisation helper for audit artifacts.
"""

import logging
from collections import deque
from typing import Dict, List, Tuple, Set, Optional

logger = logging.getLogger("GreenOps.GraphBuilder")


class DynamicGraphBuilder:
    """
    Constructs and validates the dependency graph connecting functions to tests.

    The graph maps each *changed* function to the set of tests that are
    semantically similar (above a threshold). The builder also supports
    transitive expansion via a call graph, and weighted scoring.
    """

    def __init__(self):
        self.logger = logger
        # Populated after build_heuristic_graph(); used by transitive helpers
        self._last_graph: Dict[str, List[str]] = {}
        self._last_tests: List[str] = []

    # ── Core graph builder ───────────────────────────────────────────────────

    def build_heuristic_graph(
        self,
        changed_functions:  List[str],
        similarity_scores:  Dict[Tuple[str, str], float],
        similarity_threshold: float,
    ) -> Tuple[Dict[str, List[str]], List[str]]:
        """
        Builds the static base layer of the dependency graph.

        Args:
            changed_functions:  Names of functions that were modified in the PR.
            similarity_scores:  {(function, test): cosine_similarity_score}
            similarity_threshold: Minimum score to include a (function, test) edge.

        Returns:
            (graph, impacted_tests)
            graph          : {function_name: [sorted list of impacted test names]}
            impacted_tests : sorted deduplicated list of all impacted test names
        """
        self.logger.info(
            "Building dependency graph — %d changed functions, threshold=%.2f",
            len(changed_functions), similarity_threshold,
        )

        graph: Dict[str, Set[str]] = {func: set() for func in changed_functions}
        impacted_tests: Set[str] = set()

        for (func, test), score in similarity_scores.items():
            if func in changed_functions and score >= similarity_threshold:
                graph[func].add(test)
                impacted_tests.add(test)

        # FIX: convert sets to sorted lists for determinism and correct typing
        final_graph = {k: sorted(list(v)) for k, v in graph.items()}

        self._last_graph = final_graph
        self._last_tests = sorted(list(impacted_tests))

        self.logger.info(
            "Graph built: %d functions → %d unique impacted tests",
            len(final_graph), len(self._last_tests),
        )
        return final_graph, self._last_tests

    # ── Weighted scoring ─────────────────────────────────────────────────────

    def weighted_impact_scores(
        self,
        similarity_scores: Dict[Tuple[str, str], float],
        similarity_threshold: float,
        changed_functions: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Compute a float impact weight per test.

        Weight = sum of similarity scores from all (function, test) pairs
        above threshold. Tests matched by multiple changed functions get
        higher weights and should be prioritised.

        Returns: {test_name: weight}
        """
        weights: Dict[str, float] = {}
        for (func, test), score in similarity_scores.items():
            if changed_functions and func not in changed_functions:
                continue
            if score >= similarity_threshold:
                weights[test] = weights.get(test, 0.0) + score

        return dict(sorted(weights.items(), key=lambda x: x[1], reverse=True))

    # ── Transitive expansion ─────────────────────────────────────────────────

    def reachable_tests(
        self,
        call_graph: Dict[str, List[str]],
        graph: Optional[Dict[str, List[str]]] = None,
        max_depth: int = 3,
    ) -> Dict[str, List[str]]:
        """
        Expand the dependency graph transitively using the call graph.

        For each function F that has direct impacted tests, walk the call graph
        to find functions that call F (or are called by F) up to max_depth hops.
        Their tests are also included with a depth penalty noted in the result.

        Uses BFS with cycle detection to avoid infinite loops.

        Args:
            call_graph : {caller: [callees]} from ast_parser.build_call_graph()
            graph      : base graph from build_heuristic_graph() or self._last_graph
            max_depth  : maximum BFS hops

        Returns:
            Augmented graph: {function_name: [tests]} including transitive matches
        """
        if graph is None:
            graph = self._last_graph
        if not graph:
            self.logger.warning("reachable_tests() called before build_heuristic_graph()")
            return {}

        # Build reverse call graph: callee → [callers]
        reverse_cg: Dict[str, Set[str]] = {}
        for caller, callees in call_graph.items():
            for callee in callees:
                reverse_cg.setdefault(callee, set()).add(caller)

        augmented = {k: list(v) for k, v in graph.items()}
        seed_funcs = set(graph.keys())

        # BFS
        visited: Set[str] = set(seed_funcs)
        queue   = deque((f, 0) for f in seed_funcs)

        while queue:
            func, depth = queue.popleft()
            if depth >= max_depth:
                continue

            # Expand to callers (upstream) — they transitively depend on changed code
            for upstream in reverse_cg.get(func, set()):
                if upstream not in visited:
                    visited.add(upstream)
                    # Propagate the tests reachable from func to upstream
                    existing = set(augmented.get(func, []))
                    current  = set(augmented.get(upstream, []))
                    merged   = sorted(current | existing)
                    if merged != sorted(current):
                        augmented[upstream] = merged
                    queue.append((upstream, depth + 1))

        n_added = len(augmented) - len(graph)
        self.logger.info(
            "Transitive expansion: %d direct + %d transitive functions (max_depth=%d)",
            len(graph), n_added, max_depth,
        )
        return augmented

    # ── Cycle detection ──────────────────────────────────────────────────────

    @staticmethod
    def find_cycles(call_graph: Dict[str, List[str]]) -> List[List[str]]:
        """
        Detect cycles in the call graph using DFS.
        Returns a list of cycles (each cycle is a list of function names).
        Useful for warning about recursive call chains that could inflate
        transitive test impact.
        """
        visited:    Set[str] = set()
        rec_stack:  Set[str] = set()
        cycles:     List[List[str]] = []

        def dfs(node: str, path: List[str]):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbour in call_graph.get(node, []):
                if neighbour not in visited:
                    dfs(neighbour, path)
                elif neighbour in rec_stack:
                    # Found a cycle — extract it
                    cycle_start = path.index(neighbour)
                    cycles.append(path[cycle_start:])

            path.pop()
            rec_stack.discard(node)

        for node in list(call_graph.keys()):
            if node not in visited:
                dfs(node, [])

        return cycles

    # ── Serialisation ────────────────────────────────────────────────────────

    def to_artifact(
        self,
        graph: Dict[str, List[str]],
        impacted_tests: List[str],
        similarity_scores: Optional[Dict[Tuple[str, str], float]] = None,
        similarity_threshold: float = 0.5,
    ) -> dict:
        """
        Serialise the graph to a JSON-compatible dict for artifact storage.
        """
        weights = {}
        if similarity_scores:
            weights = self.weighted_impact_scores(
                similarity_scores, similarity_threshold, list(graph.keys())
            )

        return {
            "graph":          graph,
            "impacted_tests": impacted_tests,
            "test_weights":   {t: round(w, 4) for t, w in weights.items()},
            "stats": {
                "n_changed_functions": len(graph),
                "n_impacted_tests":    len(impacted_tests),
            },
        }
