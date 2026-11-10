"""Minimal source-span mutation operators (PLAN 5.1). Edits replace an exact (l0,c0,l1,c1) span of the
original text; the file is never reformatted. Columns are character offsets (converted from ast's UTF-8 byte offsets)."""
import ast, random
from dataclasses import dataclass

CMP_FLIP = {ast.Lt: ("<", "<="), ast.LtE: ("<=", "<"), ast.Gt: (">", ">="), ast.GtE: (">=", ">"),
            ast.Eq: ("==", "!="), ast.NotEq: ("!=", "==")}
ARITH = {ast.Add: ("+", "-"), ast.Sub: ("-", "+"), ast.Mult: ("*", "//"), ast.FloorDiv: ("//", "*")}
BOOL = {ast.And: ("and", "or"), ast.Or: ("or", "and")}


@dataclass
class Mutation:
    operator: str
    span: tuple  # (l0, c0, l1, c1), 1-based lines, 0-based char cols, end exclusive
    old: str
    new: str


def _char_col(line, byte_col):
    return len(line.encode("utf-8")[:byte_col].decode("utf-8", errors="ignore"))


def node_span(lines, n):
    return (n.lineno, _char_col(lines[n.lineno - 1], n.col_offset), n.end_lineno, _char_col(lines[n.end_lineno - 1], n.end_col_offset))


def get_span(lines, span):
    l0, c0, l1, c1 = span
    if l0 == l1:
        return lines[l0 - 1][c0:c1]
    return "\n".join([lines[l0 - 1][c0:]] + lines[l0:l1 - 1] + [lines[l1 - 1][:c1]])


def apply(src, m):
    lines = src.split("\n")
    assert get_span(lines, m.span) == m.old, "span/old mismatch"
    l0, c0, l1, c1 = m.span
    head, tail = lines[l0 - 1][:c0], lines[l1 - 1][c1:]
    new_lines = (head + m.new + tail).split("\n")
    return "\n".join(lines[:l0 - 1] + new_lines + lines[l1:])


def _op_between(lines, a, b, tok):
    """Find the operator token text between end of node a and start of node b."""
    s = node_span(lines, a); e = node_span(lines, b)
    if s[2] != e[0]:
        return None  # operator on a different line from its operands' boundary: skip for safety
    seg = lines[s[2] - 1][s[3]:e[1]]
    i = seg.find(tok)
    if i < 0 or seg.count(tok) != 1:
        return None
    return (s[2], s[3] + i, s[2], s[3] + i + len(tok))


def candidates_in_function(src, fn_node, rng):
    lines = src.split("\n")
    out = []
    nested = {id(n) for n in ast.walk(fn_node) if n is not fn_node and isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef))}
    inside_nested = set()
    for n in ast.walk(fn_node):
        if id(n) in nested:
            inside_nested |= {id(x) for x in ast.walk(n)}
    body_nodes = [n for n in ast.walk(fn_node) if id(n) not in inside_nested and n is not fn_node]
    doc = fn_node.body[0] if fn_node.body and isinstance(fn_node.body[0], ast.Expr) and isinstance(getattr(fn_node.body[0], "value", None), ast.Constant) else None
    for n in body_nodes:
        if isinstance(n, ast.Compare) and len(n.ops) == 1 and type(n.ops[0]) in CMP_FLIP:
            old, new = CMP_FLIP[type(n.ops[0])]
            sp = _op_between(lines, n.left, n.comparators[0], old)
            if sp:
                out.append(Mutation("cmp_flip", sp, old, new))
        elif isinstance(n, ast.BinOp) and type(n.op) in ARITH:
            old, new = ARITH[type(n.op)]
            sp = _op_between(lines, n.left, n.right, old)
            if sp:
                out.append(Mutation("arith_swap", sp, old, new))
        elif isinstance(n, ast.BoolOp) and type(n.op) in BOOL and len(n.values) == 2:
            old, new = BOOL[type(n.op)]
            sp = _op_between(lines, n.values[0], n.values[1], f" {old} ")
            if sp:
                out.append(Mutation("bool_swap", sp, f" {old} ", f" {new} "))
        elif isinstance(n, ast.Constant) and type(n.value) is int and n is not doc:
            sp = node_span(lines, n); old = get_span(lines, sp)
            if old.isdigit():
                out.append(Mutation("off_by_one", sp, old, str(n.value + rng.choice((1, -1)) if n.value else 1)))
        elif isinstance(n, ast.Call) and len(n.args) >= 2 and not any(isinstance(a, ast.Starred) for a in n.args[:2]):
            a, b = n.args[0], n.args[1]
            sa, sb = node_span(lines, a), node_span(lines, b)
            if sa[0] == sa[2] == sb[0] == sb[2]:
                ta, tb = get_span(lines, sa), get_span(lines, sb)
                if ta != tb:
                    mid = lines[sa[0] - 1][sa[3]:sb[1]]
                    out.append(Mutation("arg_swap", (sa[0], sa[1], sb[2], sb[3]), ta + mid + tb, tb + mid + ta))
        elif isinstance(n, (ast.If, ast.While)) and isinstance(n, ast.If):
            sp = node_span(lines, n.test); old = get_span(lines, sp)
            if "\n" not in old:
                out.append(Mutation("cond_removal", sp, old, "True"))
        elif isinstance(n, ast.Return) and n.value is not None:
            sp = node_span(lines, n.value); old = get_span(lines, sp)
            new = {"None": "True", "True": "False", "False": "True"}.get(old, "None")
            out.append(Mutation("return_change", sp, old, new))
    # wrong-variable substitution: swap a loaded local name for another local loaded in the function
    names = sorted({n.id for n in body_nodes if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)} &
                   ({a.arg for a in fn_node.args.args + fn_node.args.kwonlyargs} |
                    {n.id for n in body_nodes if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}))
    if len(names) >= 2:
        for n in body_nodes:
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in names:
                alt = rng.choice([x for x in names if x != n.id])
                out.append(Mutation("wrong_var", node_span(lines, n), n.id, alt))
    # statement deletion -> pass (keeps indentation and block validity)
    for n in body_nodes:
        for field in ("body", "orelse"):
            blk = getattr(n, field, None) if n is not fn_node else None
            if isinstance(blk, list) and len(blk) >= 2:
                for s in blk:
                    if isinstance(s, (ast.Expr, ast.Assign, ast.AugAssign, ast.AnnAssign)) and s.lineno == s.end_lineno and s is not doc:
                        sp = node_span(lines, s)
                        out.append(Mutation("stmt_deletion", sp, get_span(lines, sp), "pass"))
    for s in fn_node.body[1:] if len(fn_node.body) >= 2 else []:
        if isinstance(s, (ast.Expr, ast.Assign, ast.AugAssign)) and s.lineno == s.end_lineno and s is not doc:
            sp = node_span(lines, s)
            out.append(Mutation("stmt_deletion", sp, get_span(lines, sp), "pass"))
    seen, uniq = set(), []
    for m in out:
        k = (m.operator, m.span)
        if k not in seen:
            seen.add(k); uniq.append(m)
    return uniq


def is_valid_mutant(orig_src, new_src):
    try:
        return ast.dump(ast.parse(orig_src)) != ast.dump(ast.parse(new_src))
    except SyntaxError:
        return False
