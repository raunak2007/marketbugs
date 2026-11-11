"""Prompt builders. Templates live in configs/prompts/*.txt; their hashes go into every row."""
from pathlib import Path
from src.llm.hashing import sha
from src.llm.tokens import truncate_code

PROMPT_DIR = Path(__file__).resolve().parents[2] / "configs" / "prompts"
AGENTS = ("trace", "graph", "spec", "static", "semantic")  # canonical order; bit i = 1<<i
LABELS = [chr(65 + i) for i in range(20)]


def template(name):
    return (PROMPT_DIR / f"{name}.txt").read_text()


def template_hash(name):
    return sha(template(name))


def candidate_block(view):
    """Same block for every method; code truncated to the fixed token budget (src.llm.tokens, PLAN 4.2)."""
    parts = []
    for lab, c in zip(LABELS, view["pool"]):
        code = truncate_code(c["code"])
        parts.append(f"[{lab}] {c['path']} :: {c['qualname']}\n{code}")
    return "\n\n".join(parts)


def subset_agents(mask):
    return [a for i, a in enumerate(AGENTS) if mask >> i & 1]


def probability_agent_prompt(view, notes_by_agent, mask, order=None):
    """C-mode prompt: issue, candidate block, notes in canonical order (absent omitted), answer cue
    (the cue itself is appended by the server after the chat template)."""
    present = subset_agents(mask)
    if order is not None:  # E3 duplication / B_full5 cyclic permutations only
        present = [a for a in order if a in present or a.split("#")[0] in present]
    notes = ""
    if present:
        notes = "\n\nSpecialist notes:\n" + "\n".join(f"- {a.split('#')[0]}: {notes_by_agent[a]}" for a in present)
    # issue_text already carries the traceback in the test_informed condition (and none in no_trace / SWE-bench);
    # appending view["trace"] again showed it twice (fixed pre-freeze 2026-11-10).
    issue = view["issue_text"]
    p = template("probability_agent").format(
        issue=issue, candidates=candidate_block(view), notes=notes, n=len(view["pool"]),
        last_label=LABELS[len(view["pool"]) - 1])
    return p, present
