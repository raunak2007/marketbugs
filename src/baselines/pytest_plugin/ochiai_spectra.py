"""pytest plugin for B3: per-test line coverage + outcomes. Loaded with -p inside the target repo's venv.
Writes $OCHIAI_OUT/spectra.json = {"outcomes": {nodeid: passed|failed}, "lines": {relpath: {line: [nodeid]}}}."""
import json, os
from collections import defaultdict
from pathlib import Path
import coverage

_SRC = Path(os.environ["OCHIAI_SOURCE"]).resolve()
_cov = coverage.Coverage(data_file=None, source=[str(_SRC)], omit=[str(_SRC / ".venv/*")])
_outcomes = {}


def pytest_configure(config):
    _cov.start()


def pytest_runtest_setup(item):
    _cov.switch_context(item.nodeid)


def pytest_runtest_logreport(report):
    if report.when == "call" or (report.when == "setup" and report.outcome != "passed"):
        if report.outcome in ("passed", "failed"):
            prev = _outcomes.get(report.nodeid)
            _outcomes[report.nodeid] = "failed" if "failed" in (prev, report.outcome) else "passed"


def pytest_unconfigure(config):
    _cov.stop()
    data = _cov.get_data()
    lines = defaultdict(lambda: defaultdict(list))
    for f in data.measured_files():
        try:
            rel = str(Path(f).resolve().relative_to(_SRC))
        except ValueError:
            continue
        for ln, ctxs in (data.contexts_by_lineno(f) or {}).items():
            c = [x for x in ctxs if x]
            if c:
                lines[rel][str(ln)] = c
    Path(os.environ["OCHIAI_OUT"], "spectra.json").write_text(json.dumps({"outcomes": _outcomes, "lines": lines}))
