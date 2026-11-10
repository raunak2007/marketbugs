"""Per-instance auxiliary evidence for specialist views. Computed from repo + inference view only
(never gold). Keys are pool cids."""
import ast
from src.cands.functions import trace_frames, enclosing


def static_flags(code):
    flags = []
    try:
        tree = ast.parse(code.strip() if code.lstrip() == code else __import__("textwrap").dedent(code))
    except SyntaxError:
        return ["unparseable"]
    fn = next((n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
    for n in ast.walk(tree):
        if isinstance(n, ast.Compare):
            ops = {type(o).__name__ for o in n.ops}
            if ops & {"Lt", "LtE", "Gt", "GtE"}:
                flags.append(f"boundary comparison ({'/'.join(sorted(ops))}) line {n.lineno}")
            if ops & {"Is", "IsNot"} and any(isinstance(c, ast.Constant) and not (c.value is None or isinstance(c.value, bool)) for c in n.comparators):
                flags.append(f"identity comparison with literal line {n.lineno}")
        if isinstance(n, ast.BinOp) and isinstance(n.right, ast.Constant) and n.right.value in (1, -1) and isinstance(n.op, (ast.Add, ast.Sub)):
            flags.append(f"+/-1 arithmetic line {n.lineno}")
        if isinstance(n, ast.ExceptHandler) and n.type is None:
            flags.append(f"bare except line {n.lineno}")
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for d in n.args.defaults + n.args.kw_defaults:
                if isinstance(d, (ast.List, ast.Dict, ast.Set)):
                    flags.append("mutable default argument")
        body = getattr(n, "body", None)
        if isinstance(body, list):
            for i, s in enumerate(body[:-1]):
                if isinstance(s, (ast.Return, ast.Raise, ast.Continue, ast.Break)):
                    flags.append(f"unreachable statement after line {s.lineno}"); break
    if fn is not None:
        stored = {t.id for t in ast.walk(fn) if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store)}
        loaded = {t.id for t in ast.walk(fn) if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Load)}
        unused = sorted(stored - loaded - {"_"})
        if unused:
            flags.append("assigned but unused: " + ", ".join(unused[:4]))
        params = {a.arg for a in fn.args.args + fn.args.kwonlyargs} - {"self", "cls"}
        unused_p = sorted(params - loaded)
        if unused_p:
            flags.append("unused parameter: " + ", ".join(unused_p[:4]))
    return list(dict.fromkeys(flags))[:6]


def build_evidence(view, repo_fns=None, graph=None, semantic_scores=None, repo_root=None):
    """view: inference_view(); repo_fns: Fn index; graph: retrieval.Graph; semantic_scores: {cid: float}."""
    pool = view["pool"]
    key_to_cid = {f"{c['path']}::{c['qualname']}": c["cid"] for c in pool}
    frames = []
    if repo_fns is not None and view.get("trace"):
        for path, line, name in trace_frames(view["trace"], repo_root):
            f = enclosing(repo_fns, path, line)
            frames.append({"path": path, "line": line, "fn": f.qualname if f else name,
                           "cid": key_to_cid.get(f.key) if f else None})
    neighbors = {}
    if graph is not None:
        for c in pool:
            k = f"{c['path']}::{c['qualname']}"
            nb = [key_to_cid[n] for n in sorted(graph.neighbors(k)) if n in key_to_cid]
            neighbors[c["cid"]] = nb
    return {"frames": frames, "neighbors": neighbors,
            "static": {c["cid"]: static_flags(c["code"]) for c in pool},
            "semantic": semantic_scores or {}}
