"""Placebo (irrelevant-evidence) prompts, protocol v1.0 section 3. INFERENCE SIDE.

The recipient keeps its own issue, pool, cid mapping, retrieval and candidate order; only the five note texts are
replaced by the donor's canonical notes (same specialist slot). Donor canonical notes cite pool cids "[{X}]"; they
are rendered with the same `render_note` under the RECIPIENT's view, so cited labels point at unrelated recipient
functions (an intentionally misleading control, not a guaranteed-neutral one). Same probability_agent_prompt,
mask 31, canonical note order.
"""
from src.agents.prompts import AGENTS, probability_agent_prompt
from src.agents.specialists import render_note
from src.agents.run_values import permuted_view

FULL = 31
EMPTY_NOTE = "(no usable note)"  # same fallback as the real arm (pipeline.run_instance)


def placebo_prompts(view, donor_canon, perms):
    """donor_canon: {agent: donor canonical note text}. -> [(perm, permuted view, prompt, present, rendered)]."""
    out = []
    for perm in perms:
        pv = permuted_view(view, perm)
        rn = {a: render_note(donor_canon[a] or EMPTY_NOTE, pv) for a in AGENTS}
        p, present = probability_agent_prompt(pv, rn, FULL)
        out.append((perm, pv, p, present, rn))
    return out


def real_full_prompts(view, canon, perms):
    """The genuine mask-31 canonical-order prompts, built exactly as pipeline.run_instance builds them."""
    return placebo_prompts(view, canon, perms)
