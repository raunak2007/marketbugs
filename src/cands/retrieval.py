"""Retrievers over a function index: BM25, embeddings, static call graph."""
import re
from collections import defaultdict
import numpy as np
from rank_bm25 import BM25Okapi
from .functions import trace_frames, enclosing

_TOK = re.compile(r"[A-Za-z][a-z]+|[A-Z]+(?![a-z])|\d+")


def tokenize(text):
    return [t.lower() for t in _TOK.findall(text or "")]


def fn_text(f):
    return f"{f.path} {f.qualname} {f.docstring}\n{f.code}"


class BM25:
    def __init__(self, fns):
        self.fns = fns
        self.bm = BM25Okapi([tokenize(fn_text(f)) or ["_"] for f in fns])

    def rank(self, query, k=None):
        s = self.bm.get_scores(tokenize(query) or ["_"])
        idx = np.argsort(-s, kind="stable")[:k]
        return [(self.fns[i], float(s[i])) for i in idx]


class Embedding:
    """Pluggable embedding retriever; `embed(list[str]) -> np.ndarray (n, d)` L2-normalized.
    VERIFY: model choice (default BAAI/bge-small-en-v1.5 via sentence-transformers), pinned revision."""
    def __init__(self, fns, embed):
        self.fns, self.embed = fns, embed
        self.M = embed([fn_text(f)[:2000] for f in fns]) if fns else np.zeros((0, 1))

    def rank(self, query, k=None):
        q = self.embed([query[:2000]])[0]
        s = self.M @ q
        idx = np.argsort(-s, kind="stable")[:k]
        return [(self.fns[i], float(s[i])) for i in idx]


def call_graph(fns):
    by_name = defaultdict(list)
    for f in fns:
        by_name[f.name].append(f)
    out_edges = {f.key: {g.key for c in f.calls for g in by_name.get(c, ()) if g.key != f.key} for f in fns}
    in_edges = defaultdict(set)
    for a, bs in out_edges.items():
        for b in bs:
            in_edges[b].add(a)
    return out_edges, in_edges


def frame_functions(fns, trace, repo_root=None):
    seen, out = set(), []
    for path, line, _ in trace_frames(trace, repo_root):
        f = enclosing(fns, path, line)
        if f and f.key not in seen:
            seen.add(f.key); out.append(f)
    return out


class Graph:
    def __init__(self, fns):
        self.fns = {f.key: f for f in fns}
        self.out_e, self.in_e = call_graph(fns)

    def neighbors(self, key):
        return self.out_e.get(key, set()) | self.in_e.get(key, set())

    def rank(self, frame_fns, k=None):
        """Frames first (innermost last in traceback -> rank first), then 1-hop neighbors."""
        ranked, seen = [], set()
        for f in reversed(frame_fns):
            if f.key in self.fns and f.key not in seen:
                ranked.append((self.fns[f.key], 2.0)); seen.add(f.key)
        for f in reversed(frame_fns):
            for n in sorted(self.neighbors(f.key)):
                if n not in seen:
                    ranked.append((self.fns[n], 1.0)); seen.add(n)
        return ranked[:k]
