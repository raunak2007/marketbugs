"""Function index for a checked-out repo via `ast` (no imports, no execution)."""
import ast, re, sys

sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))
SKIPPED_FILES = []  # (relpath, reason) for files that could not be indexed; reported in build logs
from dataclasses import dataclass, field
from pathlib import Path

SKIP_DIRS = {".venv", ".git", "build", "dist", "docs", "doc", "benchmarks", "node_modules", ".tox"}


@dataclass
class Fn:
    path: str
    qualname: str
    lineno: int
    end_lineno: int
    code: str
    docstring: str
    n_statements: int
    calls: set = field(default_factory=set)
    annotations: str = ""

    @property
    def name(self):
        return self.qualname.rsplit(".", 1)[-1]

    @property
    def key(self):
        return f"{self.path}::{self.qualname}"


def is_test_path(rel):
    parts = Path(rel).parts
    return any(p in ("tests", "test", "testing") for p in parts) or Path(rel).name.startswith(("test_", "conftest"))


class _V(ast.NodeVisitor):
    def __init__(self, rel, src_lines):
        self.rel, self.lines, self.stack, self.fns = rel, src_lines, [], []

    def visit_ClassDef(self, n):
        self.stack.append(n.name); self.generic_visit(n); self.stack.pop()

    def _fn(self, n):
        q = ".".join(self.stack + [n.name])
        calls = set()
        for c in ast.walk(n):
            if isinstance(c, ast.Call):
                f = c.func
                calls.add(f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None)
        calls.discard(None)
        nstmt = sum(isinstance(s, ast.stmt) for s in ast.walk(n)) - 1
        ann = ", ".join(f"{a.arg}: {ast.unparse(a.annotation)}" for a in n.args.args if a.annotation)
        if n.returns is not None:
            ann += f" -> {ast.unparse(n.returns)}"
        self.fns.append(Fn(self.rel, q, n.lineno, n.end_lineno, "\n".join(self.lines[n.lineno - 1:n.end_lineno]),
                           ast.get_docstring(n) or "", nstmt, calls, ann))
        self.stack.append(n.name); self.generic_visit(n); self.stack.pop()

    visit_FunctionDef = visit_AsyncFunctionDef = _fn


def index_repo(root, include_tests=False):
    root = Path(root); fns = []
    for p in sorted(root.rglob("*.py")):
        rel = str(p.relative_to(root))
        if set(Path(rel).parts) & SKIP_DIRS or (not include_tests and is_test_path(rel)):
            continue
        try:
            src = p.read_text(encoding="utf-8")
            tree = ast.parse(src)
        except (SyntaxError, UnicodeDecodeError, ValueError, RecursionError) as e:
            SKIPPED_FILES.append((rel, type(e).__name__)); continue
        try:
            v = _V(rel, src.splitlines()); v.visit(tree); fns.extend(v.fns)
        except RecursionError:
            SKIPPED_FILES.append((rel, "recursion"))
    return fns


def enclosing(fns, path, line):
    """Innermost function containing (path, line)."""
    best = None
    for f in fns:
        if f.path == path and f.lineno <= line <= f.end_lineno:
            if best is None or f.lineno >= best.lineno:
                best = f
    return best


FRAME_RE = re.compile(r'File "([^"]+)", line (\d+), in (\S+)')
PYTEST_FRAME_RE = re.compile(r'^([\w./-]+\.py):(\d+): ', re.M)


def trace_frames(trace, repo_root=None):
    """[(relpath, line, name)] from a Python or pytest traceback, paths made repo-relative when possible."""
    out = []
    if not trace:
        return out
    for path, line, name in FRAME_RE.findall(trace):
        out.append((_rel(path, repo_root), int(line), name))
    for path, line in PYTEST_FRAME_RE.findall(trace):
        out.append((_rel(path, repo_root), int(line), None))
    return out


def _rel(path, root):
    if root:
        try:
            return str(Path(path).resolve().relative_to(Path(root).resolve()))
        except ValueError:
            pass
    return path.lstrip("./")
