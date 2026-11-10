"""Prompt builders. Templates live in configs/prompts/*.txt; their hashes go into every row."""
from pathlib import Path
from src.llm.hashing import sha

PROMPT_DIR = Path(__file__).resolve().parents[2] / "configs" / "prompts"
AGENTS = ("trace", "graph", "spec", "static", "semantic")  # canonical order; bit i = 1<<i
LABELS = [chr(65 + i) for i in range(20)]


def template(name):
    return (PROMPT_DIR / f"{name}.txt").read_text()


def template_hash(name):
    return sha(template(name))


def candidate_block(view, code_char_budget=1200):
    # VERIFY: switch to a token budget once the tokenizer is pinned at G2 (same budget for every method).
    parts = []
    for lab, c in zip(LABELS, view["pool"]):
        code = c["code"] if len(c["code"]) <= code_char_budget else c["code"][:code_char_budget] + "\n# ...truncated"
        parts.append(f"[{lab}] {c['path']} :: {c['qualname']}\n{code}")
    return "\n\n".join(parts)


def subset_agents(mask):
    return [a for i, a in enumerate(AGENTS) if mask >> i & 1]


def probability_agent_prompt(view, notes_by_agent, mask):
    """C-mode prompt: issue, candidate block, notes in canonical order (absent omitted), answer cue
    (the cue itself is appended by the server after the chat template)."""
    present = subset_agents(mask)
    notes = ""
    if present:
        notes = "\n\nSpecialist notes:\n" + "\n".join(f"- {a}: {notes_by_agent[a]}" for a in present)
    issue = view["issue_text"] + (f"\n\nTraceback:\n{view['trace']}" if view.get("trace") else "")
    p = template("probability_agent").format(
        issue=issue, candidates=candidate_block(view), notes=notes, n=len(view["pool"]),
        last_label=LABELS[len(view["pool"]) - 1])
    return p, present
