"""Five specialists (PLAN 6). Each gets a different view + shared candidate pool.
Label conventions: prompts cite display labels as [X]; stored notes/ranks use pool cids."""
import re
from src.agents.prompts import template, LABELS, candidate_block, AGENTS
from src.cands.functions import FRAME_RE

# Gemma writes "RANKED: [F,T,A]", "RANKED: [A], [F]", "RANKED: [{L}],[{B}]" or bare "C, A,F": take every standalone
# label A-T on the RANKED line, with or without brackets/braces.
RANK_RE = re.compile(r"RANKED:[ \t]*([^\n]*)")
LABEL_RE = re.compile(r"(?<![A-Za-z])([A-T])(?![A-Za-z])")
CITE_RE = re.compile(r"\[([A-T])\]")


def label_maps(view):
    lab2cid = {LABELS[i]: c["cid"] for i, c in enumerate(view["pool"])}
    return lab2cid, {v: k for k, v in lab2cid.items()}


def _issue(view, with_trace=True):
    return view["issue_text"] + (f"\n\nTraceback:\n{view['trace']}" if with_trace and view.get("trace") else "")


def view_text(agent, view, ev):
    _, cid2lab = label_maps(view)
    L = lambda cid: f"[{cid2lab[cid]}]"
    if agent == "trace":
        if view["source"] == "swebench_lite":
            return "Issue (no traceback available):\n" + view["issue_text"]
        rows = [f"  {f['path']}:{f['line']} in {f['fn']}" + (f" -> candidate {L(f['cid'])}" if f["cid"] else " (not a candidate)") for f in ev["frames"]]
        return _issue(view, False) + "\n\nTraceback frames mapped to candidates:\n" + ("\n".join(rows) or "  (none)")
    if agent == "graph":
        fr = [f for f in ev["frames"] if f["cid"]]
        lines = [f"  frame candidate {L(f['cid'])}: neighbors " + (", ".join(L(n) for n in ev["neighbors"].get(f["cid"], [])) or "none") for f in fr]
        if not lines:
            lines = [f"  {L(c)} calls/called-by " + ", ".join(L(n) for n in nb) for c, nb in ev["neighbors"].items() if nb][:20]
        return "Static call-graph and import neighborhood of the failing code path:\n" + ("\n".join(lines) or "  (no edges)")
    if agent == "spec":
        return "Check each candidate's documented contract (docstring, type hints, comments) against what its code does."
    if agent == "static":
        rows = [f"  {L(c)}: " + ("; ".join(fl) or "no flags") for c, fl in ev["static"].items()]
        return "Automated AST pattern flags per candidate:\n" + "\n".join(rows)
    if agent == "semantic":
        sc = sorted(ev["semantic"].items(), key=lambda kv: -kv[1])
        rows = [f"  {L(c)}: similarity {s:.3f}" for c, s in sc]
        return "Embedding similarity between the bug report and each candidate:\n" + ("\n".join(rows) or "  (unavailable)")
    raise KeyError(agent)


def specialist_prompt(agent, view, ev):
    issue = view["issue_text"] if agent in ("spec", "static", "semantic", "graph") else ""
    return template("specialist").format(
        role=agent, view=view_text(agent, view, ev),
        issue=issue if agent != "trace" else "", candidates=candidate_block(view),
        last_label=LABELS[len(view["pool"]) - 1])


def forecaster_prompt(agent, view, ev):
    return template("forecaster").format(role=agent, view=view_text(agent, view, ev),
                                         issue=view["issue_text"] if agent != "trace" else "",
                                         candidates=candidate_block(view), last_label=LABELS[len(view["pool"]) - 1])


def parse_note(text, view):
    """-> (canonical_text, ranked_cids) or (None, []) if unparseable. Canonical text cites "[{cid}]"."""
    lab2cid, _ = label_maps(view)
    m = RANK_RE.search(text)
    if not m:
        return None, []
    ranked = list(dict.fromkeys(lab2cid[x] for x in LABEL_RE.findall(m.group(1)) if x in lab2cid))
    if not ranked:
        return None, []
    reasons = [l.strip() for l in text[m.end():].splitlines() if l.strip()][:3]
    reasons = [CITE_RE.sub(lambda mm: "[{" + lab2cid[mm.group(1)] + "}]" if mm.group(1) in lab2cid else mm.group(0), r) for r in reasons]
    canon = "RANKED: " + ",".join("[{" + c + "}]" for c in ranked) + ("\n" + "\n".join(reasons) if reasons else "")
    return canon, ranked


def render_note(canon, view):
    """Render a canonical note under the view's current display order."""
    _, cid2lab = label_maps(view)
    return re.sub(r"\[\{([A-T])\}\]", lambda m: f"[{cid2lab.get(m.group(1), '?')}]", canon)
