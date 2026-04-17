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
"""

import ast
import hashlib
import json
import logging
import os
import re
import tokenize
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
    calls:         list            = field(default_factory=list)  # functions this calls
    docstring:     Optional[str]   = None


@dataclass
class FileAST:
    """AST summary for a single source file."""
    file_path:     str
    language:      str
    file_hash:     str             = ""
    imports:       list            = field(default_factory=list)
    classes:       list            = field(default_factory=list)
    functions:     list            = field(default_factory=list)    # top-level
    methods:       list            = field(default_factory=list)    # inside classes
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


# ─────────────────────────────────────────────────────────────────────────────
# PYTHON AST VISITOR
# ─────────────────────────────────────────────────────────────────────────────

class PythonASTVisitor(ast.NodeVisitor):
    """
    Walks a Python AST and extracts structural features.
    Computes McCabe cyclomatic complexity per function.
    """

    # These AST nodes each add 1 to complexity
    COMPLEXITY_NODES = (
        ast.If, ast.While, ast.For, ast.ExceptHandler,
        ast.With, ast.Assert, ast.comprehension,
        ast.BoolOp,   # and/or
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

        # Compute cyclomatic complexity
        complexity = 1  # base complexity
        for child in ast.walk(node):
            if isinstance(child, self.COMPLEXITY_NODES):
                complexity += 1
            # Ternary operator in Return/Assign
            elif isinstance(child, ast.IfExp):
                complexity += 1

        # Extract function calls
        calls = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                call_name = self._name_of(child.func)
                if call_name:
                    calls.append(call_name)

        # Docstring
        docstring = ast.get_docstring(node)

        # Decorators
        has_decorator = len(node.decorator_list) > 0
        decorator_names = [self._name_of(d) for d in node.decorator_list]

        # Is this a test function?
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
            calls         = calls[:20],   # cap to avoid noise
            docstring     = docstring,
        )

        if class_name:
            self.methods.append(fn)
        else:
            self.functions.append(fn)

        self.generic_visit(node)

    def visit_Assign(self, node):
        # Capture module-level global variable names
        if not self._class_stack:
            for target in node.targets:
                name = self._name_of(target)
                if name and name.isupper():  # convention: UPPER = global constant
                    self.global_vars.append(name)
        self.generic_visit(node)

    @staticmethod
    def _name_of(node) -> str:
        """Safely extract a string name from an AST node."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return f"{PythonASTVisitor._name_of(node.value)}.{node.attr}"
        if isinstance(node, ast.Constant):
            return str(node.value)
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# JAVA AST PARSER (using javalang)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_java(source_code: str, file_path: str) -> FileAST:
    """
    Parse a Java source file using the javalang library.
    Falls back to regex extraction if javalang is unavailable.
    """
    file_ast = FileAST(file_path=file_path, language="java")

    if not JAVA_SUPPORT:
        log.debug("javalang unavailable, using regex fallback for %s", file_path)
        return _parse_java_regex(source_code, file_path)

    try:
        tree = javalang.parse.parse(source_code)

        # Imports
        file_ast.imports = [imp.path for imp in tree.imports]

        # Classes
        for _, cls in tree.filter(javalang.tree.ClassDeclaration):
            class_info = {
                "name":   cls.name,
                "bases":  [e.name for e in (cls.extends or [])] if cls.extends else [],
                "interfaces": [i.name for i in (cls.implements or [])] if cls.implements else [],
            }
            file_ast.classes.append(class_info)

            # Methods within class
            for method in cls.methods:
                complexity = 1 + sum(
                    1 for node in method.body or []
                    if isinstance(node, (
                        javalang.tree.IfStatement,
                        javalang.tree.WhileStatement,
                        javalang.tree.ForStatement,
                        javalang.tree.DoStatement,
                        javalang.tree.CatchClause,
                    ))
                )
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
        log.warning("Java parse failed for %s: %s", file_path, e)

    return file_ast


def _parse_java_regex(source_code: str, file_path: str) -> FileAST:
    """Lightweight regex-based Java parser (fallback when javalang unavailable)."""
    file_ast = FileAST(file_path=file_path, language="java")

    file_ast.imports = re.findall(r"import\s+([\w.]+);", source_code)
    file_ast.classes = [
        {"name": m} for m in re.findall(r"class\s+(\w+)", source_code)
    ]
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
def parse_file(filepath: str, repo_root: str = "."):
    """Module-level entry point for dynamic integration."""
    parser = ASTParser(repo_root=repo_root)
    # Convert FileAST object to dict for JSON compatibility in integration
    result = parser.parse_file(filepath)
    return result.to_dict()

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
        self._cache: dict = {}   # file_path → FileAST (avoid reparsing unchanged files)

    def parse_file(self, file_path: str) -> FileAST:
        """Parse a single source file and return its FileAST."""
        p = self.repo_root / file_path if not Path(file_path).is_absolute() else Path(file_path)

        if not p.exists():
            log.warning("File not found: %s", p)
            return FileAST(file_path=str(p), language="unknown",
                           parse_success=False, error_msg="File not found")

        ext = p.suffix.lower()
        language = self.SUPPORTED_EXTENSIONS.get(ext, "unknown")

        try:
            source_code = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return FileAST(file_path=str(p), language=language,
                           parse_success=False, error_msg=str(e))

        # File hash for change detection (used to skip re-parsing unchanged files)
        file_hash = hashlib.md5(source_code.encode()).hexdigest()

        # Cache check
        if file_path in self._cache and self._cache[file_path].file_hash == file_hash:
            return self._cache[file_path]

        # Dispatch to language-specific parser
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
        self._cache[file_path] = result
        return result

    def _parse_python(self, source_code: str, file_path: str) -> FileAST:
        """Parse Python source code using the built-in ast module."""
        file_ast = FileAST(file_path=file_path, language="python")
        try:
            tree    = ast.parse(source_code)
            visitor = PythonASTVisitor(file_path)
            visitor.visit(tree)

            file_ast.imports   = visitor.imports
            file_ast.classes   = visitor.classes
            file_ast.functions = visitor.functions
            file_ast.methods   = visitor.methods
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

        pattern  = "**/*" if recursive else "*"
        results  = []
        exts     = set(self.SUPPORTED_EXTENSIONS.keys())

        for p in d.glob(pattern):
            if p.suffix.lower() in exts and p.is_file():
                result = self.parse_file(str(p))
                results.append(result)

        log.info("Parsed %d files in %s", len(results), d)
        return results

    def build_call_graph(self, file_asts: list) -> dict:
        """
        Build a cross-file call graph from parsed AST nodes.
        Returns: { 'caller_func': ['callee_func_1', 'callee_func_2', ...] }
        """
        call_graph = {}

        # Index all known functions by name
        all_funcs = {}
        for file_ast in file_asts:
            for fn in file_ast.functions + file_ast.methods:
                key = fn.name if not fn.class_name else f"{fn.class_name}.{fn.name}"
                all_funcs[fn.name] = key  # simple name → qualified name

        # Build edges
        for file_ast in file_asts:
            for fn in file_ast.functions + file_ast.methods:
                caller = fn.name if not fn.class_name else f"{fn.class_name}.{fn.name}"
                callees = [
                    all_funcs.get(c.split(".")[-1], c)
                    for c in fn.calls
                    if c.split(".")[-1] in all_funcs
                ]
                if callees:
                    call_graph[caller] = callees

        log.info("Call graph built: %d nodes, %d edges",
                 len(call_graph), sum(len(v) for v in call_graph.values()))
        return call_graph

    def get_changed_functions(
        self,
        diff_text: str,
        file_path: str,
    ) -> list:
        """
        Given a unified diff string and the file's AST, return the names of
        functions/methods that were changed.

        This is the NER component: mapping code changes to functional units.
        """
        # Extract changed line numbers from diff
        changed_lines = set()
        current_line = 0
        for line in diff_text.split("\n"):
            m = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
            if m:
                current_line = int(m.group(1))
                continue
            if line.startswith("+") and not line.startswith("+++"):
                changed_lines.add(current_line)
                current_line += 1
            elif not line.startswith("-"):
                current_line += 1

        # Find functions whose line spans overlap with changed lines
        file_ast = self.parse_file(file_path)
        changed_functions = []

        for fn in file_ast.functions + file_ast.methods:
            fn_lines = set(range(fn.start_line, fn.end_line + 1))
            if fn_lines & changed_lines:
                changed_functions.append(fn.name)

        return changed_functions

    def save_ast_features(self, file_asts: list, output_path: str):
        """Save parsed AST features as JSON for downstream use."""
        data = [f.to_dict() for f in file_asts]
        with open(output_path, "w") as fp:
            json.dump(data, fp, indent=2)
        log.info("AST features saved → %s (%d files)", output_path, len(data))


# ─────────────────────────────────────────────────────────────────────────────
# TEST NAME → SOURCE FILE MAPPER
# ─────────────────────────────────────────────────────────────────────────────

def parse_test_mapping(test_names: list, repo_root: str = ".") -> dict:
    """
    Given test names from the CSV (e.g. from your dataset's test_name column),
    attempt to map each to a source file path in the repo.

    Your test names appear to be fully qualified paths like:
        //4e560f4d05/8c21ea298f/a64d32f263/...

    This function:
      1. Strips build hash prefixes to isolate the test identifier
      2. Searches for matching files in the repo
      3. Returns { test_name → file_path }

    NOTE: Without access to the actual repo, this returns the extracted
    test identifier. Plug in your actual repo_root for real file resolution.
    """
    repo = Path(repo_root)
    mapping = {}

    for test_name in test_names:
        # Strip leading // and hash components (your test names are build-qualified)
        # Example: "//4e560f4d05/.../some/package/TestClass#testMethod"
        clean = re.sub(r"^//[\w/]+\s+", "", test_name).strip()

        # Try to extract a Java-style qualified class name
        java_match = re.search(r"([\w]+(?:\.[\w]+)+)", clean)
        if java_match:
            qualified = java_match.group(1)
            # Convert com.example.TestClass → com/example/TestClass.java
            as_path = qualified.replace(".", "/") + ".java"
            candidate = repo / "src" / as_path
            if candidate.exists():
                mapping[test_name] = str(candidate)
                continue

        # Try Python-style test_module.TestClass.test_method
        py_match = re.search(r"([\w]+(?:_[\w]+)*)", clean)
        if py_match:
            module_name = py_match.group(1)
            for candidate in repo.rglob(f"{module_name}.py"):
                mapping[test_name] = str(candidate)
                break

        # If nothing found, store the cleaned name for manual mapping
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

    print(f"\nAST parsing complete:")
    print(f"  {len(file_asts)} files parsed")
    print(f"  {len(call_graph)} call graph nodes")
    print(f"  Output: {out.resolve()}")
