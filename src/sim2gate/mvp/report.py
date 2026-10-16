"""The human-readable report of one verdict (MVP plan, "Report and verdict": JSON for machines, HTML for people).

    python -m sim2gate.mvp.report verdict.json --out report.html

Built from the verdict JSON alone (sim2gate.mvp.verdict), as one self-contained HTML file. Each risk lists the check,
metric, measured value, threshold, signed margin, evidence (the sweep table or the measured values), likely cause
and suggested fix; the run's provenance (checkpoint, configuration and input hashes, versions) closes the page. The
exit code is the verdict's: 0 go, 1 no-go, 2 missing inputs, 3 check invalid.
"""
import argparse
import html
import json
import os
import sys

from ..checks import robustness

TITLES = {"2": "Check 2: simulator exploits", "1": "Check 1: robustness envelope",
          "5": "Check 5: PhysX vs Newton transfer"}
VERDICT_TEXT = {"go": "GO", "no-go": "NO-GO", "invalid": "CHECK INVALID", "incomplete": "INCOMPLETE"}
NOTE = ("Simulation warning signs, not proof of how the policy behaves on a real robot. Sim2Gate's checks are "
        "still being validated.")


def _e(x):
    return html.escape("" if x is None else str(x))


def _num(x, digits=4):
    if x is None:
        return "–"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int, float)):
        return f"{x:.{digits}g}"
    return _e(x)


def _rate(x):
    return "–" if x is None else f"{100 * x:.0f}%"


def _bar(x):
    if x is None:
        return "–"
    w = max(0.0, min(1.0, x)) * 100
    cls = "ok" if x >= robustness.CLIFF_SUCCESS else "bad"
    return (f'<span class="bar"><span class="fill {cls}" style="width:{w:.0f}%"></span></span> '
            f'<span class="pct">{_rate(x)}</span>')


def _risks(v):
    if not v["risks"]:
        return "<p>No risk flagged.</p>"
    rows = []
    for r in v["risks"]:
        rows.append(f"<tr><td>{_e(r.get('check'))}</td><td>{_e(r.get('metric'))}</td><td>{_num(r.get('value'))}</td>"
                    f"<td>{_num(r.get('threshold'))}</td><td>{_num(r.get('margin'))}</td>"
                    f"<td>{_e(r.get('likely_cause'))}</td><td>{_e(r.get('fix'))}</td></tr>")
    return ("<table><thead><tr><th>Check</th><th>Metric</th><th>Measured</th><th>Threshold</th>"
            "<th>Margin</th><th>Likely cause</th><th>Suggested fix</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>"
            "<p class='small'>Margin, checks 2 and 5: how far the measured value is from the limit; negative means past "
            "it. Check 1: how far inside the realistic range the cliff lies (the threshold column is the range's "
            "edge); any cliff inside the range is flagged.</p>")


def _status(c):
    if c is None:
        return "<p>Not run.</p>"
    s = c.get("status")
    flag = c.get("flagged")
    head = {"scored": "flagged" if flag else "passed", "invalid": "invalid", "missing": "missing"}.get(s, s)
    out = f"<p class='status {_e(head)}'>{_e(head.upper())}</p>"
    if c.get("reasons"):
        out += "<ul>" + "".join(f"<li>{_e(r)}</li>" for r in c["reasons"]) + "</ul>"
    return out


def _check2(d):
    m = (d or {}).get("metrics") or {}
    if not m:
        return ""
    rows = "".join(f"<tr class='{'flag' if x.get('flagged') else ''}'><td>{_e(k)}</td><td>{_num(x.get('value'))}</td>"
                   f"<td>{_num(x.get('threshold'))}</td><td>{_num(x.get('margin'))}</td>"
                   f"<td>{'yes' if x.get('flagged') else 'no'}</td></tr>" for k, x in m.items())
    return ("<table><thead><tr><th>Metric</th><th>Measured</th><th>Threshold</th><th>Margin</th><th>Flagged</th>"
            "</tr></thead><tbody>" + rows + "</tbody></table>")


def _training(tr):
    if not isinstance(tr, dict):
        return "not recorded" if tr is None else _e(tr)
    parts = [f"{_e(k)} {_e(v)}" for k, v in tr.items() if v is not None and k not in ("operation", "distribution")]
    return ", ".join(parts) if parts else "not found in the saved training settings"


def _check1(d):
    params = (d or {}).get("params") or {}
    out = []
    for k, p in params.items():
        series = p.get("series") or []
        cells = "".join(f"<tr class='{'cliff' if p.get('cliff') is not None and s.get('value') == p.get('cliff') else ''}'>"
                        f"<td>{_num(s.get('value'))}</td><td>{_bar(s.get('success_rate'))}</td>"
                        f"<td>{_e(s.get('episodes'))}</td></tr>" for s in series)
        lo, hi = (p.get("plausible") or [None, None])
        out.append(f"<h4>{_e(k)} ({_e(p.get('unit'))}){' — FLAGGED' if p.get('flagged') else ''}</h4>"
                   f"<p class='small'>Realistic range {_num(lo)}–{_num(hi)}; cliff (first value under "
                   f"{_rate(robustness.CLIFF_SUCCESS)} success): {_num(p.get('cliff'))}; training range: "
                   f"{_training(p.get('training_range'))}</p>"
                   "<table class='sweep'><thead><tr><th>Value</th><th>Episode success</th><th>Episodes</th></tr>"
                   "</thead><tbody>" + cells + "</tbody></table>")
    return "".join(out)


def _check5(d):
    m = (d or {}).get("metrics") or {}
    if not m:
        return ""
    rule = (d or {}).get("rule") or {}
    rows = "".join(f"<tr><td>{_e(k)}</td><td>{_num(x)}</td></tr>" for k, x in m.items())
    return (f"<p class='small'>Parity: {_e((d or {}).get('parity'))}. Flagged if fewer than "
            f"{_rate(rule.get('newton_fall_free_min'))} of Newton episodes are fall-free, or the Newton/PhysX "
            f"tracking-error ratio exceeds {_num(rule.get('error_ratio_max'))}.</p>"
            "<table><tbody>" + rows + "</tbody></table>")


def _joint(j, problems):
    if j is None and not problems:
        return ""
    out = ["<h2>Joint sweep (report only, never part of the verdict)</h2>"]
    if problems:
        out.append("<ul>" + "".join(f"<li>{_e(p)}</li>" for p in problems) + "</ul>")
    if j and j.get("status") == "scored":
        ar, ab = j["all_realistic"], j["any_beyond"]
        out.append(f"<p>All five conditions drawn together. Success with every condition inside its realistic "
                   f"range at once: <b>{_rate(ar['success_rate'])}</b> of {ar['episodes']} episodes; with any "
                   f"condition beyond it: {_rate(ab['success_rate'])} of {ab['episodes']}.</p>")
        rows = "".join(f"<tr><td>{_e(' + '.join(p['pair']))}</td><td>{_rate(p['both_hard'])}</td>"
                       f"<td>{_rate(p.get('expected_if_independent'))}</td><td>{_num(p.get('interaction'), 2)}</td></tr>"
                       for p in j.get("worst_pairs") or [])
        if rows:
            out.append("<p class='small'>Pairs that fail more together than apart (both in the harder half of their "
                       "realistic range):</p><table><thead><tr><th>Pair</th><th>Success, both hard</th>"
                       "<th>Expected if independent</th><th>Difference</th></tr></thead><tbody>" + rows
                       + "</tbody></table>")
    elif j:
        out.append(_status(j))
    return "".join(out)


def _provenance(v):
    prov = v.get("provenance") or {}
    rows = [("Policy ID", v["policy_id"]), ("Checkpoint SHA-256", v.get("checkpoint_sha256")),
            ("Sim2Gate", prov.get("sim2gate_version")),
            ("Simulator versions", json.dumps(prov.get("simulator_versions"), sort_keys=True)),
            ("Task (as run)", (prov.get("sweep") or {}).get("task")),
            ("Environment config SHA-256", (prov.get("sweep") or {}).get("env_cfg_sha256")),
            ("Command set SHA-256", (prov.get("sweep") or {}).get("command_set_sha256")),
            ("Seed", (prov.get("sweep") or {}).get("seed")),
            ("Overrides", " ".join((prov.get("sweep") or {}).get("overrides") or []))]
    for k, h in (v.get("frozen") or {}).items():
        rows.append((k.replace("_", " "), h))
    for k, h in (v.get("inputs_sha256") or {}).items():
        rows.append((f"input {k}", h))
    for k, m in (v.get("measurements") or {}).items():
        rows.append((f"measurement {k}", f"{m.get('path')} ({m.get('sha256')})"))
    return "<table class='prov'><tbody>" + "".join(f"<tr><td>{_e(a)}</td><td><code>{_e(b)}</code></td></tr>"
                                                   for a, b in rows) + "</tbody></table>"


CSS = """
:root{--bg:#fff;--fg:#1a1a1a;--muted:#5b5b5b;--line:#d9d9d9;--ok:#2e7d32;--bad:#c62828;--warn:#b26a00;--card:#f6f6f6}
@media (prefers-color-scheme: dark){:root{--bg:#141414;--fg:#ececec;--muted:#a8a8a8;--line:#3a3a3a;--ok:#66bb6a;
--bad:#ef5350;--warn:#ffb74d;--card:#1f1f1f}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
max-width:1000px;margin:0 auto;padding:16px}
h1{font-size:22px;margin:8px 0}h2{font-size:18px;margin-top:28px;border-bottom:1px solid var(--line)}
h4{margin:16px 0 4px}.small{color:var(--muted);font-size:13px}
.verdict{padding:12px 16px;border-radius:8px;background:var(--card);border-left:6px solid var(--muted);font-size:20px}
.verdict.go{border-color:var(--ok)}.verdict.no-go{border-color:var(--bad)}.verdict.invalid,.verdict.incomplete{border-color:var(--warn)}
table{border-collapse:collapse;width:100%;margin:6px 0 12px;font-size:14px}
td,th{border-bottom:1px solid var(--line);padding:5px 8px;text-align:left;vertical-align:top}
tr.flag td,tr.cliff td{font-weight:600}
.status.flagged{color:var(--bad);font-weight:700}.status.passed{color:var(--ok);font-weight:700}
.status.invalid,.status.missing{color:var(--warn);font-weight:700}
.bar{display:inline-block;width:120px;height:10px;background:var(--line);border-radius:5px;vertical-align:middle}
.fill{display:block;height:100%;border-radius:5px}.fill.ok{background:var(--ok)}.fill.bad{background:var(--bad)}
.pct{display:inline-block;min-width:40px}code{font-size:12px;word-break:break-all}
.prov td:first-child{width:28%;color:var(--muted)}
@media (max-width:640px){table{display:block;overflow-x:auto}}
"""


def render(v):
    verdict_cls = _e(v["verdict"])
    parts = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
             f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>Sim2Gate report {_e(v['policy_id'])}</title><style>{CSS}</style></head><body>",
             f"<h1>Sim2Gate report: {_e(v['policy_id'])}</h1>",
             f"<div class='verdict {verdict_cls}'><b>{_e(VERDICT_TEXT.get(v['verdict'], v['verdict']))}</b> "
             f"<span class='small'>(exit code {_e(v['exit_code'])})</span></div>",
             f"<p class='small'>{_e(NOTE)}</p>"]
    if v.get("problems"):
        parts.append("<h2>Problems</h2><ul>" + "".join(f"<li>{_e(p)}</li>" for p in v["problems"]) + "</ul>")
    parts.append("<h2>Risks</h2>" + _risks(v))
    det = v.get("details") or {}
    for name in ("2", "1", "5"):
        parts.append(f"<h2>{_e(TITLES[name])}</h2>" + _status(v["checks"].get(name)))
        parts.append({"2": _check2, "1": _check1, "5": _check5}[name](det.get(name)))
    ro = v.get("report_only") or {}
    parts.append(_joint(ro.get("joint_sweep"), ro.get("problems")))
    parts.append("<h2>Provenance</h2>" + _provenance(v))
    parts.append("</body></html>")
    return "\n".join(parts)


def write(v, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(render(v))
    os.replace(tmp, path)
    return path


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m sim2gate.mvp.report", description=__doc__.splitlines()[0])
    p.add_argument("verdict", help="a verdict JSON (sim2gate.mvp.verdict)")
    p.add_argument("--out", required=True, help="the HTML file to write")
    a = p.parse_args(argv)
    try:
        v = json.load(open(a.verdict))
    except (OSError, ValueError) as e:
        print(f"[sim2gate] cannot read {a.verdict}: {e}", file=sys.stderr)
        return 2
    write(v, a.out)
    print(f"[sim2gate] {v['policy_id']}: {v['verdict']} -> {a.out}")
    return int(v["exit_code"])


if __name__ == "__main__":
    sys.exit(main())
