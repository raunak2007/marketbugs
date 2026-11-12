"""Placebo donor assignment (protocol v1.0 section 3). Pure function of instance IDs + seed: never sees gold, outcomes,
notes or mutation metadata, so changing any gold label cannot change an assignment.

Per repository: sort IDs, shuffle with random.Random(f"{seed}|{repo}"), then donor(L[i]) = L[(i+1) % n]. A cyclic
shift of a shuffled list has no fixed points for n >= 2, so termination is guaranteed without rejection sampling.
One donor instance per recipient supplies all five specialist notes (same specialist slot -> same specialist).
"""
import hashlib, json, random

PLACEBO_SEED = 20261111


def derangement(ids, seed, repo):
    ids = sorted(set(ids))
    if len(ids) < 2:
        raise ValueError(f"repo {repo}: need >= 2 instances for a derangement, got {len(ids)}")
    order = list(ids)
    random.Random(f"{seed}|{repo}").shuffle(order)
    return {order[i]: order[(i + 1) % len(order)] for i in range(len(order))}


def assign_donors(id_repo_pairs, seed=PLACEBO_SEED):
    """id_repo_pairs: iterable of (instance_id, repo). Returns {recipient_id: donor_id}."""
    by = {}
    for i, r in id_repo_pairs:
        by.setdefault(r, []).append(i)
    out = {}
    for r in sorted(by):
        out.update(derangement(by[r], seed, r))
    return out


def assignment_hash(assign):
    return hashlib.sha256(json.dumps(sorted(assign.items())).encode()).hexdigest()


def sha256(s):
    return hashlib.sha256(s.encode() if isinstance(s, str) else s).hexdigest()
