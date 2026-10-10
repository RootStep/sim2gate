import os, sys
sys.path.insert(0, os.path.dirname(__file__))


def pytest_sessionfinish(session, exitstatus):
    """With SIM2GATE_ALLOWED_SKIPS set (comma-separated nodeid fragments), any other skipped test fails the run,
    so CI cannot silently skip an acceptance test."""
    allowed = os.environ.get("SIM2GATE_ALLOWED_SKIPS")
    if allowed is None:
        return
    tr = session.config.pluginmanager.get_plugin("terminalreporter")
    skipped = tr.stats.get("skipped", []) if tr is not None else []
    frags = [a.strip() for a in allowed.split(",") if a.strip()]
    bad = [r.nodeid for r in skipped if not any(f in r.nodeid for f in frags)]
    if bad:
        print("\nunexpected skipped tests: " + ", ".join(bad))
        session.exitstatus = 1
