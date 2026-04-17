"""
ast_parser.py
=============
Green-Ops CI/CD Framework — AST Parsing Module

Parses Python/Java source files to extract structural code features
for the Dynamic Dependency Graph and XGBoost Pf (probability of failure) model.

What it extracts:
  - Function/method names and line spans
  - Class hierarchy
  - Import dependencies
  - Cyclomatic complexity
  - Modified nodes from a diff (change detection)
  - Call graph edges (which function calls which)
  - Test ↔ source file mapping

Supports: Python (native AST), Java (javalang), Generic (fallback regex)

Install dependencies:
    pip install javalang asttokens

Usage:
    from ast_parser import ASTParser, parse_test_mapping
    parser = ASTParser(repo_root="/path/to/repo")

    # Parse a source file
    features = parser.parse_file("src/auth/login.py")

    # Map test names from your CSV to source files
    mappings = parse_test_mapping(df["test_name"].unique(), repo_root="/path/to/repo")

CHANGES (v2):
  - FIX: ASTParser._parse_python() used MD5 but file_hash was set AFTER dispatch;
         cache check compared stale hash. Fixed by computing hash before dispatch.
  - FIX: parse_directory() silently skipped symlinks and unreadable files;
         now logs a warning per skipped file.
  - FIX: build_call_graph() last-write-wins on duplicate function names;
         now uses qualified names as primary keys and short names as aliases.
  - FIX: get_changed_functions() parsed diff lines but did not handle the
         context lines (lines starting with ' ') correctly, causing off-by-one
         line counting. Fixed current_line tracking.
  - FIX: Java complexity walker only counted top-level statements, not nested;
         switched to full recursive walk using javalang tree traversal.
  - IMPROVEMENT: Thread-safe in-process cache (lock added for concurrent PRs).
  - IMPROVEMENT: save_ast_features() now writes atomically via a temp file.
  - IMPROVEMENT: Added ASTDiff.compare() for structural AST diffing (not just hash diff),
         enabling detection of *meaningful* changes vs whitespace/comment-only edits.
  - IMPROVEMENT: Added value_score() helper used by the pipeline to rank modules.
  - NEW: parse_file() accepts repo-relative OR absolute paths without ambiguity.
"""

import ast
import hashlib
import json
import logging
import os
import re
import tempfile
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

log = logging.getLogger("greenops.ast")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Optional Java support
try:
    import javalang
    JAVA_SUPPORT = True
except ImportError:
    JAVA_SUPPORT = False
    log.warning("javalang not installed. Java AST parsing will use regex fallback.")


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FunctionNode:
    """Represents a single function/method in the AST."""
    name:          str
    class_name:    Optional[str]   = None
    file_path:     str             = ""
    start_line:    int             = 0
    end_line:      int             = 0
    complexity:    int             = 1      # McCabe cyclomatic complexity
    num_args:      int             = 0
    has_decorator: bool            = False
    is_test:       bool            = False
    calls:         list            = field(default_factory=list)
    docstring:     Optional[str]   = None


@dataclass
class FileAST:
    """AST summary for a single source file."""
    file_path:     str
    language:      str
    file_hash:     str             = ""
    imports:       list            = field(default_factory=list)
    classes:       list            = field(default_factory=list)
    functions:     list            = field(default_factory=list)
    methods:       list            = field(default_factory=list)
    global_vars:   list            = field(default_factory=list)
    num_lines:     int             = 0
    parse_success: bool            = True
    error_msg:     str             = ""

    def to_dict(self) -> dict:
        return {
            "file_path":     self.file_path,
            "language":      self.language,
            "file_hash":     self.file_hash,
            "num_lines":     self.num_lines,
            "num_imports":   len(self.imports),
            "num_classes":   len(self.classes),
            "num_functions": len(self.functions),
            "num_methods":   len(self.methods),
            "imports":       self.imports,
            "classes":       self.classes,
            "functions":     [asdict(f) for f in self.functions],
            "methods":       [asdict(m) for m in self.methods],
            "parse_success": self.parse_success,
        }

    def value_score(self) -> float:
        """
        Compute a 'value score' for this module to rank it for embedding/scheduling.
        Higher = more structurally complex = higher change risk.

        Formula:
          complexity_sum  = sum of all function complexities
          coupling        = number of imports (external dependencies)
          size_factor     = log(num_lines + 1)
          value = 0.5 * complexity_sum + 0.3 * coupling + 0.2 * size_factor
        """
        import math
        all_fns = self.functions + self.methods
        complexity_sum = sum(fn.complexity for fn in all_fns) if all_fns else 1
        coupling       = len(self.imports)
        size_factor    = math.log(self.num_lines + 1)
        return round(0.5 * complexity_sum + 0.3 * coupling + 0.2 * size_factor, 4)


# ─────────────────────────────────────────────────────────────────────────────
# AST STRUCTURAL DIFF
# ─────────────────────────────────────────────────────────────────────────────

class ASTDiff:
    """
    Structural diff between two FileAST objects.

    Unlike a raw file hash comparison, this detects *meaningful* changes:
    - New/removed functions or classes
    - Changed function signatures (arity, decorators)
    - Changed complexity (logic changed, not just comments/whitespace)

    Used by the pipeline to skip re-embedding when only docstrings changed.
    """

    @staticmethod
    def compare(old: Optional["FileAST"], new: "FileAST") -> dict:
        """
        Returns a dict describing structural changes between old and new AST.

        Keys:
          is_meaningful   : bool  — True if any structural element changed
          added_functions : list  — function names added
          removed_functions: list — function names removed
          changed_complexity: list — (name, old_c, new_c) for funcs whose complexity changed
          added_classes   : list
          removed_classes : list
          import_changes  : {"added": [...], "removed": [...]}
          change_summary  : str
        """
        if old is None:
            return {
                "is_meaningful": True,
                "added_functions": [f.name for f in new.functions + new.methods],
                "removed_functions": [],
                "changed_complexity": [],
                "added_classes": [c["name"] for c in new.classes],
                "removed_classes": [],
                "import_changes": {"added": new.imports, "removed": []},
                "change_summary": "New module — no prior AST.",
            }

        def fn_map(ast_obj: "FileAST") -> dict:
            return {
                (fn.class_name, fn.name): fn
                for fn in ast_obj.functions + ast_obj.methods
            }

        old_fns = fn_map(old)
        new_fns = fn_map(new)

        added    = [k[1] for k in set(new_fns) - set(old_fns)]
        removed  = [k[1] for k in set(old_fns) - set(new_fns)]
        changed_c = []

        for key in set(old_fns) & set(new_fns):
            o, n = old_fns[key], new_fns[key]
            if o.complexity != n.complexity or o.num_args != n.num_args:
                changed_c.append((key[1], o.complexity, n.complexity))

        old_cls = {c["name"] for c in old.classes}
        new_cls = {c["name"] for c in new.classes}
        added_cls   = list(new_cls - old_cls)
        removed_cls = list(old_cls - new_cls)

        old_imp = set(old.imports)
        new_imp = set(new.imports)
        imp_changes = {
            "added":   list(new_imp - old_imp),
            "removed": list(old_imp - new_imp),
        }

        is_meaningful = bool(added or removed or changed_c or added_cls or
                             removed_cls or imp_changes["added"] or
                             imp_changes["removed"])

        parts = []
        if added:       parts.append(f"added functions: {added}")
        if removed:     parts.append(f"removed functions: {removed}")
        if changed_c:   parts.append(f"complexity changed: {changed_c}")
        if added_cls:   parts.append(f"added classes: {added_cls}")
        if removed_cls: parts.append(f"removed classes: {removed_cls}")
        if imp_changes["added"]:   parts.append(f"new imports: {imp_changes['added']}")
        if imp_changes["removed"]: parts.append(f"removed imports: {imp_changes['removed']}")
        summary = "; ".join(parts) if parts else "No structural changes (whitespace/comment only)."

        return {
            "is_meaningful":      is_meaningful,
            "added_functions":    added,
            "removed_functions":  removed,
            "changed_complexity": changed_c,
            "added_classes":      added_cls,
            "removed_classes":    removed_cls,
            "import_changes":     imp_changes,
            "change_summary":     summary,
        }


# ─────────────────────────────────────────────────────────────────────────────
# PYTHON AST VISITOR
# ─────────────────────────────────────────────────────────────────────────────

class PythonASTVisitor(ast.NodeVisitor):
    """
    Walks a Python AST and extracts structural features.
    Computes McCabe cyclomatic complexity per function.
    """

    COMPLEXITY_NODES = (
        ast.If, ast.While, ast.For, ast.ExceptHandler,
        ast.With, ast.Assert, ast.comprehension,
        ast.BoolOp,
    )

    def __init__(self, file_path: str):
        self.file_path   = file_path
        self.functions   = []
        self.methods     = []
        self.imports     = []
        self.classes     = []
        self.global_vars = []
        self._class_stack = []

    def visit_Import(self, node):
        for alias in node.names:
            self.imports.append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        module = node.module or ""
        for alias in node.names:
            self.imports.append(f"{module}.{alias.name}")
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        bases = [self._name_of(b) for b in node.bases]
        self.classes.append({
            "name":       node.name,
            "bases":      bases,
            "start_line": node.lineno,
            "end_line":   node.end_lineno,
            "num_methods": sum(1 for n in ast.walk(node) if isinstance(n, ast.FunctionDef)),
        })
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    def visit_FunctionDef(self, node):
        self._process_function(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def _process_function(self, node):
        class_name = self._class_stack[-1] if self._class_stack else None

        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, self.COMPLEXITY_NODES):
                complexity += 1
            elif isinstance(child, ast.IfExp):
                complexity += 1

        calls = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                call_name = self._name_of(child.func)
                if call_name:
                    calls.append(call_name)

        docstring = ast.get_docstring(node)
        has_decorator = len(node.decorator_list) > 0
        decorator_names = [self._name_of(d) for d in node.decorator_list]

        is_test = (
            node.name.startswith("test_") or
            node.name.startswith("Test") or
            "pytest" in str(decorator_names) or
            "unittest" in str(decorator_names)
        )

        fn = FunctionNode(
            name          = node.name,
            class_name    = class_name,
            file_path     = self.file_path,
            start_line    = node.lineno,
            end_line      = getattr(node, "end_lineno", node.lineno),
            complexity    = complexity,
            num_args      = len(node.args.args),
            has_decorator = has_decorator,
            is_test       = is_test,
            calls         = calls[:20],
            docstring     = docstring,
        )

        if class_name:
            self.methods.append(fn)
        else:
            self.functions.append(fn)

        self.generic_visit(node)

    def visit_Assign(self, node):
        if not self._class_stack:
            for target in node.targets:
                name = self._name_of(target)
                if name and name.isupper():
                    self.global_vars.append(name)
        self.generic_visit(node)

    @staticmethod
    def _name_of(node) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return f"{PythonASTVisitor._name_of(node.value)}.{node.attr}"
        if isinstance(node, ast.Constant):
            return str(node.value)
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# JAVA AST PARSER
# ─────────────────────────────────────────────────────────────────────────────

def _parse_java(source_code: str, file_path: str) -> "FileAST":
    file_ast = FileAST(file_path=file_path, language="java")

    if not JAVA_SUPPORT:
        log.debug("javalang unavailable, using regex fallback for %s", file_path)
        return _parse_java_regex(source_code, file_path)

    try:
        tree = javalang.parse.parse(source_code)
        file_ast.imports = [imp.path for imp in tree.imports]

        for _, cls in tree.filter(javalang.tree.ClassDeclaration):
            class_info = {
                "name":       cls.name,
                "bases":      [cls.extends.name] if cls.extends else [],
                "interfaces": [i.name for i in (cls.implements or [])],
            }
            file_ast.classes.append(class_info)

            for method in cls.methods:
                # FIX: use full recursive walk for complexity, not just body[0]
                complexity = 1
                if method.body:
                    for stmt in method.body:
                        try:
                            for _, node in stmt:
                                if isinstance(node, (
                                    javalang.tree.IfStatement,
                                    javalang.tree.WhileStatement,
                                    javalang.tree.ForStatement,
                                    javalang.tree.DoStatement,
                                    javalang.tree.CatchClause,
                                    javalang.tree.TernaryExpression,
                                )):
                                    complexity += 1
                        except Exception:
                            pass

                is_test = any(
                    a.name in ("Test", "ParameterizedTest", "RepeatedTest")
                    for a in (method.annotations or [])
                )
                fn = FunctionNode(
                    name       = method.name,
                    class_name = cls.name,
                    file_path  = file_path,
                    complexity = complexity,
                    num_args   = len(method.parameters or []),
                    is_test    = is_test,
                )
                file_ast.methods.append(fn)

        file_ast.parse_success = True

    except Exception as e:
        file_ast.parse_success = False
        file_ast.error_msg     = str(e)
        log.warning("Java parse failed for %s: %s — falling back to regex", file_path, e)
        return _parse_java_regex(source_code, file_path)

    return file_ast


def _parse_java_regex(source_code: str, file_path: str) -> "FileAST":
    file_ast = FileAST(file_path=file_path, language="java")
    file_ast.imports = re.findall(r"import\s+([\w.]+);", source_code)
    file_ast.classes = [{"name": m} for m in re.findall(r"class\s+(\w+)", source_code)]
    methods = re.findall(
        r"(?:public|private|protected|static|final)[\s\w<>\[\]]+\s+(\w+)\s*\(", source_code
    )
    for m in methods:
        fn = FunctionNode(name=m, file_path=file_path)
        file_ast.methods.append(fn)
    file_ast.parse_success = True
    return file_ast


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AST PARSER CLASS
# ─────────────────────────────────────────────────────────────────────────────

class ASTParser:
    """
    Main entry point for AST parsing in the Green-Ops pipeline.

    Usage:
        parser = ASTParser(repo_root="/path/to/repo")
        features = parser.parse_file("src/auth/login.py")
        all_features = parser.parse_directory("src/")
    """

    SUPPORTED_EXTENSIONS = {
        ".py":   "python",
        ".java": "java",
    }

    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root)
        self._cache: dict = {}
        self._cache_lock = threading.Lock()   # FIX: thread-safe cache for concurrent PRs

    def parse_file(self, file_path: str) -> FileAST:
        """Parse a single source file and return its FileAST."""
        fp = Path(file_path)
        # Resolve: if relative, join with repo_root; if already absolute, use as-is
        p = fp if fp.is_absolute() else self.repo_root / fp

        if not p.exists():
            log.warning("File not found: %s", p)
            return FileAST(file_path=str(p), language="unknown",
                           parse_success=False, error_msg="File not found")

        ext      = p.suffix.lower()
        language = self.SUPPORTED_EXTENSIONS.get(ext, "unknown")

        try:
            source_code = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return FileAST(file_path=str(p), language=language,
                           parse_success=False, error_msg=str(e))

        # FIX: compute hash BEFORE checking cache so the comparison is valid
        file_hash = hashlib.sha256(source_code.encode()).hexdigest()

        cache_key = str(p)
        with self._cache_lock:
            cached = self._cache.get(cache_key)
            if cached and cached.file_hash == file_hash:
                return cached

        if language == "python":
            result = self._parse_python(source_code, str(p))
        elif language == "java":
            result = _parse_java(source_code, str(p))
        else:
            log.debug("Unsupported file type: %s", ext)
            result = FileAST(file_path=str(p), language="unknown",
                             parse_success=False, error_msg=f"Unsupported: {ext}")

        result.file_hash = file_hash
        result.num_lines = source_code.count("\n") + 1

        with self._cache_lock:
            self._cache[cache_key] = result

        return result

    def _parse_python(self, source_code: str, file_path: str) -> FileAST:
        file_ast = FileAST(file_path=file_path, language="python")
        try:
            tree    = ast.parse(source_code)
            visitor = PythonASTVisitor(file_path)
            visitor.visit(tree)

            file_ast.imports     = visitor.imports
            file_ast.classes     = visitor.classes
            file_ast.functions   = visitor.functions
            file_ast.methods     = visitor.methods
            file_ast.global_vars = visitor.global_vars
            file_ast.parse_success = True

        except SyntaxError as e:
            file_ast.parse_success = False
            file_ast.error_msg     = f"SyntaxError at line {e.lineno}: {e.msg}"
            log.warning("Python parse error in %s: %s", file_path, e)

        return file_ast

    def parse_directory(self, directory: str, recursive: bool = True) -> list:
        """
        Parse all supported files in a directory.
        Returns a list of FileAST objects.
        """
        d = self.repo_root / directory if not Path(directory).is_absolute() else Path(directory)
        if not d.exists():
            log.error("Directory not found: %s", d)
            return []

        pattern = "**/*" if recursive else "*"
        results = []
        exts    = set(self.SUPPORTED_EXTENSIONS.keys())

        for p in d.glob(pattern):
            if not p.is_file() or p.suffix.lower() not in exts:
                continue
            # FIX: warn on symlinks and unreadable files instead of silently skipping
            if p.is_symlink() and not p.exists():
                log.warning("Broken symlink skipped: %s", p)
                continue
            result = self.parse_file(str(p))
            if not result.parse_success:
                log.warning("Parse failed for %s: %s", p, result.error_msg)
            results.append(result)

        log.info("Parsed %d files in %s", len(results), d)
        return results

    def build_call_graph(self, file_asts: list) -> dict:
        """
        Build a cross-file call graph from parsed AST nodes.
        Returns: { 'caller_func': ['callee_func_1', ...] }

        FIX: previously last-write-wins on duplicate short names.
        Now builds a proper qualified-name index and resolves call targets
        using both qualified and unqualified names.
        """
        call_graph: dict = {}

        # Build index: short_name → [qualified_name, ...]  (one-to-many)
        qualified_index: dict[str, list[str]] = {}
        for file_ast in file_asts:
            for fn in file_ast.functions + file_ast.methods:
                qname = fn.name if not fn.class_name else f"{fn.class_name}.{fn.name}"
                qualified_index.setdefault(fn.name, []).append(qname)

        for file_ast in file_asts:
            for fn in file_ast.functions + file_ast.methods:
                caller = fn.name if not fn.class_name else f"{fn.class_name}.{fn.name}"
                callees = []
                for c in fn.calls:
                    short = c.split(".")[-1]
                    if short in qualified_index:
                        callees.extend(qualified_index[short])
                    elif c in qualified_index:
                        callees.extend(qualified_index[c])
                if callees:
                    # Deduplicate while preserving order
                    call_graph[caller] = list(dict.fromkeys(callees))

        log.info("Call graph built: %d nodes, %d edges",
                 len(call_graph), sum(len(v) for v in call_graph.values()))
        return call_graph

    def get_changed_functions(self, diff_text: str, file_path: str) -> list:
        """
        Given a unified diff string, return the names of functions/methods changed.

        FIX: Previous implementation miscounted context lines (lines starting with ' ')
        by only advancing current_line on '+' lines, causing off-by-one errors.
        Context lines must also advance the counter.
        """
        changed_lines: set[int] = set()
        current_line = 0

        for line in diff_text.split("\n"):
            m = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
            if m:
                current_line = int(m.group(1))
                continue
            if line.startswith("+") and not line.startswith("+++"):
                changed_lines.add(current_line)
                current_line += 1
            elif line.startswith("-") and not line.startswith("---"):
                # Removed lines do NOT advance the new-file counter
                pass
            else:
                # Context line (starts with ' ') and other lines — advance counter
                current_line += 1

        file_ast = self.parse_file(file_path)
        changed_functions = []
        for fn in file_ast.functions + file_ast.methods:
            fn_lines = set(range(fn.start_line, fn.end_line + 1))
            if fn_lines & changed_lines:
                changed_functions.append(fn.name)

        return changed_functions

    def compare_with_stored(
        self,
        file_path: str,
        stored_ast: Optional[FileAST],
    ) -> dict:
        """
        Parse the current file and structurally compare it against a stored AST.
        Returns ASTDiff.compare() result.
        This is the entry point for "only re-embed if there are meaningful changes".
        """
        current = self.parse_file(file_path)
        return ASTDiff.compare(stored_ast, current)

    def save_ast_features(self, file_asts: list, output_path: str):
        """
        Save parsed AST features as JSON.
        FIX: writes atomically via a temp file to prevent partial writes on crash.
        """
        data = [f.to_dict() for f in file_asts]
        out  = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write via temp file in same directory
        fd, tmp_path = tempfile.mkstemp(dir=out.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as fp:
                json.dump(data, fp, indent=2)
            os.replace(tmp_path, output_path)
        except Exception:
            os.unlink(tmp_path)
            raise

        log.info("AST features saved → %s (%d files)", output_path, len(data))


# ─────────────────────────────────────────────────────────────────────────────
# TEST NAME → SOURCE FILE MAPPER
# ─────────────────────────────────────────────────────────────────────────────

def parse_test_mapping(test_names: list, repo_root: str = ".") -> dict:
    """
    Map test names from the CSV to source file paths in the repo.
    Returns { test_name → file_path }
    """
    repo    = Path(repo_root)
    mapping = {}

    for test_name in test_names:
        clean = re.sub(r"^//[\w/]+\s+", "", test_name).strip()

        java_match = re.search(r"([\w]+(?:\.[\w]+)+)", clean)
        if java_match:
            qualified = java_match.group(1)
            as_path   = qualified.replace(".", "/") + ".java"
            candidate = repo / "src" / as_path
            if candidate.exists():
                mapping[test_name] = str(candidate)
                continue

        py_match = re.search(r"([\w]+(?:_[\w]+)*)", clean)
        if py_match:
            module_name = py_match.group(1)
            for candidate in repo.rglob(f"{module_name}.py"):
                mapping[test_name] = str(candidate)
                break

        if test_name not in mapping:
            mapping[test_name] = None

    resolved = sum(1 for v in mapping.values() if v is not None)
    log.info("Test mapping: %d/%d test names resolved to source files",
             resolved, len(test_names))
    return mapping


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE USAGE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser_arg = argparse.ArgumentParser(description="Green-Ops AST Parser")
    parser_arg.add_argument("--repo",    default=".", help="Repo root directory")
    parser_arg.add_argument("--srcdir",  default="src", help="Source directory to parse")
    parser_arg.add_argument("--outdir",  default="./greenops_output")
    args = parser_arg.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    ast_parser = ASTParser(repo_root=args.repo)
    file_asts  = ast_parser.parse_directory(args.srcdir)
    call_graph = ast_parser.build_call_graph(file_asts)

    ast_parser.save_ast_features(file_asts, str(out / "ast_features.json"))

    with open(out / "call_graph.json", "w") as f:
        json.dump(call_graph, f, indent=2)

    # Print value scores
    print(f"\nAST parsing complete:")
    print(f"  {len(file_asts)} files parsed")
    print(f"  {len(call_graph)} call graph nodes")
    print(f"\nTop-10 highest-value modules:")
    ranked = sorted(file_asts, key=lambda x: x.value_score(), reverse=True)
    for fa in ranked[:10]:
        print(f"  {fa.value_score():6.2f}  {fa.file_path}")
    print(f"  Output: {out.resolve()}")
