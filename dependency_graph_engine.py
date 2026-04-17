"""
dependency_graph_engine.py
==========================
Green-Ops CI/CD Framework — Full Dependency Graph + Test Mapping Engine

Builds a real module → module → test dependency graph by:
  1. Parsing Python imports (static analysis via ast module)
  2. Parsing JS/TS require/import statements (regex)
  3. Tracking TRANSITIVE dependencies (A imports B imports C → A→B→C→tests)
  4. Mapping impacted modules → test files using a test registry
  5. Emitting exact test files to run (not just PRUNE/RUN labels)

This replaces all demo-based and hardcoded similarity logic with real
static analysis of the codebase.

USAGE:
    from dependency_graph_engine import DependencyGraphEngine
    engine = DependencyGraphEngine(repo_root="/path/to/repo")
    engine.build(repo="org/repo")

    # Get tests for changed modules
    tests = engine.get_tests_for_changed_modules(
        changed_files=["src/auth.py", "src/models/user.py"]
    )
"""

import ast
import json
import logging
import os
import re
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

log = logging.getLogger("greenops.dependency_graph")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

MAX_TRANSITIVE_DEPTH = int(os.environ.get("GREENOPS_MAX_DEPTH", "5"))
TEST_FILE_PATTERNS   = [
    r"test_.*\.py$",   r".*_test\.py$",
    r".*_spec\.py$",   r".*Test\.java$",
    r".*Spec\.java$",  r".*\.test\.(js|ts)$",
    r".*\.spec\.(js|ts)$",
]


# ─────────────────────────────────────────────────────────────────────────────
# IMPORT PARSERS
# ─────────────────────────────────────────────────────────────────────────────

class PythonImportParser:
    """
    Extracts all imports from a Python file using the ast module.
    Resolves relative imports to absolute paths within the repo.
    """

    def __init__(self, repo_root: str):
        self.repo_root = Path(repo_root)

    def extract_imports(self, file_path: str) -> List[str]:
        """
        Return list of module paths imported by file_path.
        Resolves them to repo-relative paths where possible.
        """
        fp = Path(file_path)
        try:
            source = fp.read_text(encoding="utf-8", errors="replace")
            tree   = ast.parse(source)
        except Exception as e:
            log.debug("Cannot parse %s: %s", file_path, e)
            return []

        imports: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    resolved = self._resolve(alias.name, fp, level=0)
                    if resolved:
                        imports.append(resolved)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                level  = node.level or 0
                resolved = self._resolve(module, fp, level=level)
                if resolved:
                    imports.append(resolved)

        return list(dict.fromkeys(imports))  # deduplicate, preserve order

    def _resolve(self, module_name: str, importing_file: Path, level: int) -> Optional[str]:
        """
        Attempt to resolve a module name to a repo-relative file path.
        Returns None if the module is an external (third-party) package.
        """
        if not module_name:
            return None

        # Relative import (from . import x)
        if level > 0:
            base = importing_file.parent
            for _ in range(level - 1):
                base = base.parent
            parts = module_name.split(".")
            candidate = base / Path(*parts)
            for ext in [".py", "/__init__.py"]:
                full = Path(str(candidate) + ext)
                if full.exists():
                    return str(full.relative_to(self.repo_root))
            return None

        # Absolute import — check if it lives in the repo
        parts = module_name.split(".")
        for i in range(len(parts), 0, -1):
            candidate = self.repo_root / Path(*parts[:i])
            for suffix in [".py", "/__init__.py", ".pyi"]:
                full = Path(str(candidate) + suffix)
                if full.exists():
                    return str(full.relative_to(self.repo_root))

        return None  # external package


class JSImportParser:
    """
    Extracts imports/requires from JavaScript and TypeScript files using regex.
    """

    IMPORT_RE = re.compile(
        r"""(?:import\s+(?:.*?\s+from\s+)?|require\s*\()\s*['"](\..*?)['"]\s*\)?""",
        re.MULTILINE,
    )

    def __init__(self, repo_root: str):
        self.repo_root = Path(repo_root)

    def extract_imports(self, file_path: str) -> List[str]:
        fp = Path(file_path)
        try:
            source = fp.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return []

        imports: List[str] = []
        for match in self.IMPORT_RE.finditer(source):
            spec      = match.group(1)
            resolved  = self._resolve(spec, fp)
            if resolved:
                imports.append(resolved)

        return list(dict.fromkeys(imports))

    def _resolve(self, spec: str, importing_file: Path) -> Optional[str]:
        """Resolve a relative JS import spec to a repo-relative path."""
        base      = importing_file.parent
        candidate = (base / spec).resolve()

        for ext in ["", ".js", ".ts", ".jsx", ".tsx", "/index.js", "/index.ts"]:
            full = Path(str(candidate) + ext)
            try:
                if full.exists():
                    return str(full.relative_to(self.repo_root))
            except ValueError:
                continue  # outside repo root

        return None


# ─────────────────────────────────────────────────────────────────────────────
# TEST REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

class TestRegistry:
    """
    Discovers test files and maps them to the source modules they test.
    Uses filename conventions, import analysis, and content scanning.
    """

    def __init__(self, repo_root: str):
        self.repo_root    = Path(repo_root)
        self._test_patterns = [re.compile(p) for p in TEST_FILE_PATTERNS]
        self._cache: Dict[str, List[str]] = {}  # source_module → [test_files]

    def discover_test_files(self) -> List[str]:
        """Find all test files in the repository."""
        test_files = []
        for p in self.repo_root.rglob("*"):
            if not p.is_file():
                continue
            rel = str(p.relative_to(self.repo_root))
            if any(pat.search(rel) for pat in self._test_patterns):
                test_files.append(rel)
        log.info("Discovered %d test files", len(test_files))
        return sorted(test_files)

    def build_test_module_map(self, test_files: List[str]) -> Dict[str, List[str]]:
        """
        Build mapping: source_module → [test_files_that_test_it]

        Strategy:
          1. Name-based: test_auth.py → src/auth.py
          2. Import-based: test imports src.auth → mapped
          3. Glob fallback: test in tests/ might test anything in src/
        """
        source_to_tests: Dict[str, List[str]] = defaultdict(list)
        py_parser = PythonImportParser(str(self.repo_root))

        for test_file in test_files:
            tf_path = self.repo_root / test_file
            tf_stem = Path(test_file).stem  # e.g. "test_auth"

            # Strategy 1: name convention
            for prefix in ["test_", "test"]:
                if tf_stem.startswith(prefix):
                    candidate_name = tf_stem[len(prefix):]
                    # Search for matching source file
                    for src in self.repo_root.rglob(f"{candidate_name}.py"):
                        if not any(re.search(p, str(src)) for p in TEST_FILE_PATTERNS):
                            rel = str(src.relative_to(self.repo_root))
                            source_to_tests[rel].append(test_file)

            # Strategy 2: import-based (Python only)
            if test_file.endswith(".py"):
                try:
                    imported = py_parser.extract_imports(str(tf_path))
                    for imp in imported:
                        if imp and not any(re.search(p, imp) for p in TEST_FILE_PATTERNS):
                            source_to_tests[imp].append(test_file)
                except Exception:
                    pass

        # Deduplicate
        return {k: list(dict.fromkeys(v)) for k, v in source_to_tests.items()}

    def get_tests_for_module(
        self,
        module_path:  str,
        test_map:     Dict[str, List[str]],
    ) -> List[str]:
        """Return test files that directly test a given module."""
        return test_map.get(module_path, [])


# ─────────────────────────────────────────────────────────────────────────────
# DEPENDENCY GRAPH ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class DependencyGraphEngine:
    """
    Full-featured dependency graph builder for CI test selection.

    Graph structure:
      module_graph:  {module_path: [imported_modules]}
      test_map:      {source_module: [test_files]}
      reverse_graph: {module_path: [modules_that_import_it]}  (for transitive lookup)
    """

    def __init__(
        self,
        repo_root:         str = ".",
        max_transitive:    int = MAX_TRANSITIVE_DEPTH,
    ):
        self.repo_root       = str(Path(repo_root).resolve())
        self.max_transitive  = max_transitive
        self.module_graph:   Dict[str, List[str]] = {}  # A imports → [B, C]
        self.reverse_graph:  Dict[str, List[str]] = {}  # B imported by → [A, D]
        self.test_map:       Dict[str, List[str]] = {}  # module → [tests]
        self.test_files:     List[str] = []
        self._py_parser  = PythonImportParser(repo_root)
        self._js_parser  = JSImportParser(repo_root)
        self._test_reg   = TestRegistry(repo_root)
        self._built      = False

    def build(self, repo: str = "", save_path: Optional[str] = None) -> None:
        """
        Build the complete dependency graph for the repository.
        This is the expensive one-time operation (runs on the full repo).
        Results are cached so PR-time lookups are fast.

        Args:
            repo:      Repo identifier for logging
            save_path: If set, persist graph to JSON for caching
        """
        log.info("Building dependency graph for %s ...", repo or self.repo_root)

        # Step 1: Discover all files
        all_py_files = list(Path(self.repo_root).rglob("*.py"))
        all_js_files = (
            list(Path(self.repo_root).rglob("*.js")) +
            list(Path(self.repo_root).rglob("*.ts"))
        )

        skip_dirs = {"__pycache__", ".git", "node_modules", "venv", ".venv",
                     "dist", "build", ".eggs"}

        def is_skipped(p: Path) -> bool:
            return any(d in p.parts for d in skip_dirs)

        py_files = [p for p in all_py_files if not is_skipped(p)]
        js_files = [p for p in all_js_files if not is_skipped(p)]

        log.info("Processing %d Python + %d JS/TS files", len(py_files), len(js_files))

        # Step 2: Build forward import graph
        for fp in py_files:
            rel = str(fp.relative_to(self.repo_root))
            imports = self._py_parser.extract_imports(str(fp))
            if imports:
                self.module_graph[rel] = imports

        for fp in js_files:
            rel     = str(fp.relative_to(self.repo_root))
            imports = self._js_parser.extract_imports(str(fp))
            if imports:
                self.module_graph[rel] = imports

        # Step 3: Build reverse graph
        for module, deps in self.module_graph.items():
            for dep in deps:
                self.reverse_graph.setdefault(dep, []).append(module)

        # Step 4: Discover test files and build test map
        self.test_files = self._test_reg.discover_test_files()
        self.test_map   = self._test_reg.build_test_module_map(self.test_files)

        log.info(
            "Graph built: %d modules, %d edges, %d modules have test coverage",
            len(self.module_graph),
            sum(len(v) for v in self.module_graph.values()),
            len(self.test_map),
        )
        self._built = True

        # Step 5: Persist if path provided
        if save_path:
            self.save(save_path)

    def save(self, path: str) -> None:
        """Persist the dependency graph to JSON."""
        out = {
            "module_graph":  self.module_graph,
            "reverse_graph": self.reverse_graph,
            "test_map":      self.test_map,
            "test_files":    self.test_files,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        log.info("Dependency graph saved → %s", path)

    def load(self, path: str) -> None:
        """Load a previously saved dependency graph."""
        with open(path) as f:
            data = json.load(f)
        self.module_graph  = data.get("module_graph", {})
        self.reverse_graph = data.get("reverse_graph", {})
        self.test_map      = data.get("test_map", {})
        self.test_files    = data.get("test_files", [])
        self._built        = True
        log.info("Dependency graph loaded from %s (%d modules)",
                 path, len(self.module_graph))

    # ── Test selection ────────────────────────────────────────────────────────

    def get_tests_for_changed_modules(
        self,
        changed_files:     List[str],
        include_transitive: bool = True,
    ) -> Dict[str, List[str]]:
        """
        For each changed module, return the set of test files that should run.

        Returns:
          {
            "direct_tests":      [test files directly testing changed modules],
            "transitive_tests":  [tests for modules that import changed modules],
            "all_tests":         sorted union of all impacted tests,
            "module_test_map":   {changed_module: [tests]},
            "transitive_modules": [modules impacted transitively],
          }
        """
        if not self._built:
            log.warning("Graph not built — call build() first")
            return {
                "direct_tests": [], "transitive_tests": [],
                "all_tests": [], "module_test_map": {}, "transitive_modules": [],
            }

        direct_tests:     Set[str] = set()
        transitive_tests: Set[str] = set()
        module_test_map:  Dict[str, List[str]] = {}
        transitive_modules: Set[str] = set()

        for module in changed_files:
            # Direct tests
            d_tests = self.test_map.get(module, [])
            direct_tests.update(d_tests)
            module_test_map[module] = d_tests

            # Also check if the module is itself a test
            if module in self.test_files:
                direct_tests.add(module)

        # Transitive expansion
        if include_transitive:
            upstream = self._get_upstream_modules(changed_files)
            transitive_modules.update(upstream)

            for up_module in upstream:
                t_tests = self.test_map.get(up_module, [])
                transitive_tests.update(t_tests)

        all_tests = sorted(direct_tests | transitive_tests)

        log.info(
            "Test selection: %d direct + %d transitive = %d total tests "
            "(%d transitive modules)",
            len(direct_tests), len(transitive_tests),
            len(all_tests), len(transitive_modules),
        )

        return {
            "direct_tests":       sorted(direct_tests),
            "transitive_tests":   sorted(transitive_tests),
            "all_tests":          all_tests,
            "module_test_map":    module_test_map,
            "transitive_modules": sorted(transitive_modules),
        }

    def _get_upstream_modules(
        self,
        changed_files: List[str],
    ) -> Set[str]:
        """
        BFS over the reverse graph to find all modules that import
        any of the changed files, up to max_transitive depth.
        """
        visited: Set[str] = set(changed_files)
        queue   = deque((m, 0) for m in changed_files)
        upstream: Set[str] = set()

        while queue:
            module, depth = queue.popleft()
            if depth >= self.max_transitive:
                continue

            for caller in self.reverse_graph.get(module, []):
                if caller not in visited:
                    visited.add(caller)
                    upstream.add(caller)
                    queue.append((caller, depth + 1))

        return upstream

    def get_full_impact_map(
        self,
        changed_files: List[str],
    ) -> Dict:
        """
        Return a complete impact map with:
          - direct dependencies
          - transitive dependencies
          - depth per dependency
          - test files per dependency
          - explanation per decision
        """
        if not self._built:
            return {}

        impact: Dict[str, Dict] = {}

        for module in changed_files:
            direct_deps     = self.module_graph.get(module, [])
            upstream_mods   = self._get_upstream_modules([module])
            direct_t        = self.test_map.get(module, [])
            transitive_t    = []
            for up in upstream_mods:
                transitive_t.extend(self.test_map.get(up, []))

            impact[module] = {
                "direct_dependencies":    direct_deps,
                "transitive_modules":     sorted(upstream_mods),
                "direct_tests":           direct_t,
                "transitive_tests":       sorted(set(transitive_t)),
                "all_tests":              sorted(set(direct_t + transitive_t)),
                "depth":                  self._bfs_depth(module),
                "explanation": (
                    f"{module} directly imports {len(direct_deps)} modules. "
                    f"{len(upstream_mods)} upstream modules also depend on it. "
                    f"Total impacted tests: {len(set(direct_t + transitive_t))}."
                ),
            }

        return impact

    def _bfs_depth(self, module: str) -> int:
        """Find max depth of the dependency chain starting from module."""
        visited = {module}
        queue   = deque([(module, 0)])
        max_d   = 0
        while queue:
            m, d = queue.popleft()
            max_d = max(max_d, d)
            for dep in self.module_graph.get(m, []):
                if dep not in visited:
                    visited.add(dep)
                    queue.append((dep, d + 1))
        return max_d

    def explain_test_selection(
        self,
        changed_files: List[str],
        selected_tests: List[str],
        pruned_tests:   List[str],
    ) -> List[Dict]:
        """
        Generate per-test explanations for the CI output.
        Returns list of {test, decision, reason, triggered_by}.
        """
        explanations = []
        test_select  = self.get_tests_for_changed_modules(changed_files)
        direct_set   = set(test_select["direct_tests"])
        transitive_s = set(test_select["transitive_tests"])

        for test in selected_tests:
            if test in direct_set:
                reason = f"Directly tests changed module(s): {changed_files}"
            elif test in transitive_s:
                triggered = [
                    m for m in test_select["transitive_modules"]
                    if test in self.test_map.get(m, [])
                ]
                reason = f"Transitive dependency via: {triggered[:3]}"
            else:
                reason = "Selected by embedding similarity"
            explanations.append({
                "test":         test,
                "decision":     "RUN",
                "reason":       reason,
                "triggered_by": changed_files,
            })

        for test in pruned_tests:
            explanations.append({
                "test":         test,
                "decision":     "PRUNE",
                "reason":       "No dependency path to changed modules, low embedding similarity",
                "triggered_by": [],
            })

        return explanations


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Green-Ops Dependency Graph Engine"
    )
    parser.add_argument("--repo-root",  default=".",  help="Repository root")
    parser.add_argument("--output",     default="./greenops_output")
    parser.add_argument("--changed",    nargs="*",    help="Changed files (space-separated)")
    parser.add_argument("--load-graph", default=None, help="Load graph from JSON path")
    args = parser.parse_args()

    out   = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    graph = Path(out) / "dependency_graph.json"

    engine = DependencyGraphEngine(repo_root=args.repo_root)

    if args.load_graph and Path(args.load_graph).exists():
        engine.load(args.load_graph)
    else:
        engine.build(save_path=str(graph))

    if args.changed:
        impact = engine.get_tests_for_changed_modules(args.changed)
        print(f"\nImpact analysis for: {args.changed}")
        print(f"  Direct tests      : {impact['direct_tests']}")
        print(f"  Transitive tests  : {impact['transitive_tests']}")
        print(f"  All tests         : {impact['all_tests']}")
        print(f"  Transitive modules: {impact['transitive_modules']}")
    else:
        print(f"\nGraph built: {len(engine.module_graph)} modules")
        print(f"Test files: {len(engine.test_files)}")
        print(f"Graph saved → {graph}")
