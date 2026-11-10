from src.swe.build_lite import gold_functions, stratified_sample
from src.cands.functions import Fn

PATCH = """diff --git a/pkg/m.py b/pkg/m.py
--- a/pkg/m.py
+++ b/pkg/m.py
@@ -3,4 +3,4 @@ def f(x):
     a = 1
-    return x + a
+    return x - a
 
 def g():
@@ -10,2 +10,3 @@ def h():
     y = 2
+    z = 3
     return y
"""


def test_gold_mapping_pre_patch_lines_and_insertions():
    fns = [Fn("pkg/m.py", "f", 2, 4, "", "", 2), Fn("pkg/m.py", "g", 6, 8, "", "", 2), Fn("pkg/m.py", "h", 9, 12, "", "", 2)]
    keys, files, why = gold_functions(PATCH, fns)
    assert keys == ["pkg/m.py::f", "pkg/m.py::h"] and files == ["pkg/m.py"] and why is None


def test_sample_is_seeded_and_sized():
    rows = [{"repo": f"r{i % 3}", "instance_id": f"r{i % 3}-{i}"} for i in range(90)] + [{"repo": "tiny", "instance_id": "tiny-1"}]
    a, q = stratified_sample(rows, 20, 1)
    b, _ = stratified_sample(rows, 20, 1)
    assert len(a) == 20 and a == b and q["tiny"] == 1
