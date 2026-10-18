"""v2 phase 0A driver (sim2gate.v2.phase0a): commands, record checks, resumable training, bundle. No Isaac Lab."""
import json
import os
import tarfile

import pytest

from sim2gate.v2 import phase0a


def test_plan_is_ten_runs():
    assert len(phase0a.runs()) == 10
    assert {a for a, _ in phase0a.runs()} == {"base", "no_slip", "no_action_rate", "bsrs_0.5", "bsrs_2"}


def test_train_command():
    cmd = phase0a.train_command("bsrs_2", 1, "w")
    i = cmd.index("--")
    ours, rest = cmd[:i], cmd[i + 1:]
    assert ours[ours.index("--eta") + 1] == "2.0"
    assert ours[ours.index("--evidence") + 1] == os.path.join("w", "runs", "bsrs_2_s1", "evidence")
    assert "--evidence-refit-terms" not in ours
    assert rest[rest.index("--task") + 1] == "Sim2Gate-Velocity-Flat-Go2-P0-v4"
    assert rest[rest.index("--max_iterations") + 1] == "2000"
    assert "sim2gate.v2.phase0a.register" in rest
    deep = [int(x) for x in ours[ours.index("--evidence-deep") + 1].split(",")]
    assert deep[0] == 0 and deep[-1] == 1999 and deep == sorted(set(deep))
    base = phase0a.train_command("base", 1, "w")
    assert base[base.index("--evidence-refit-terms") + 1] == "feet_slide,action_rate_l2"
    assert "--evidence-refit-terms" not in phase0a.train_command("base", 2, "w")
    short = phase0a.train_command("no_slip", 2, "w", iterations=12)
    assert short[short.index("--task") + 1] == "Sim2Gate-V2-0A-Go2-NoSlip-v1"
    assert max(int(x) for x in short[short.index("--evidence-deep") + 1].split(",")) < 12


def test_outcome_commands_use_the_arm_task():
    cmds, files = phase0a.outcome_commands("no_action_rate", "/x/model_1999.pt", "o")
    for c in cmds.values():
        assert c[c.index("--task") + 1] == "Sim2Gate-V2-0A-Go2-NoActionRate-v1"
        assert "/x/model_1999.pt" in c
    assert set(files) == {"check2", "nominal"}


def test_check_record_on_a_real_record(tmp_path):
    pytest.importorskip("rsl_rl")
    from test_evidence import train, ITERATIONS
    train(4, 0.0, tmp_path)
    assert phase0a.check_record(str(tmp_path), ITERATIONS) == []
    assert phase0a.check_record(str(tmp_path), ITERATIONS + 1) != []
    os.remove(tmp_path / "deep" / "it00001_samples.npz")
    assert "sample dump 1 missing" in phase0a.check_record(str(tmp_path), ITERATIONS)


def test_train_all_resumes_and_bundle(tmp_path, monkeypatch):
    calls = []

    def fake_train(arm, seed, work, runner, **kw):
        calls.append((arm, seed))
        d = os.path.join(work, "runs", phase0a.run_name(arm, seed), "evidence")
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "updates.jsonl"), "w").write("{}\n")
        ok = (arm, seed) != ("no_slip", 2) or len(calls) > 10
        return {"arm": arm, "seed": seed, "ok": ok, "problems": [] if ok else ["x"]}
    monkeypatch.setattr(phase0a, "_train", fake_train)
    work = str(tmp_path / "w")
    assert phase0a.train_all(work, None) == 1
    assert len(calls) == 10
    assert phase0a.train_all(work, None) == 0          # only the failed run again
    assert calls[10:] == [("no_slip", 2)]
    out = str(tmp_path / "b.tgz")
    assert phase0a.bundle(work, out) == 0
    with tarfile.open(out) as t:
        names = t.getnames()
        man = json.load(t.extractfile("phase0a/MANIFEST.json"))
    assert "phase0a/runs/base_s1/evidence/updates.jsonl" in names
    assert "phase0a/FORMAT.md" in names
    assert len(man["files"]) == 11


def test_launcher_evidence_flags():
    import argparse
    from sim2gate.training import isaaclab_launch as L
    ns = argparse.Namespace(evidence="e", evidence_sample_envs=8, evidence_refit_terms="a, b",
                            evidence_refit_iterations="1,5", evidence_deep="0,3", evidence_grad_save="3", eta=0.5)
    kw = L._evidence_kwargs(ns, ["--task", "T"])
    assert kw["out_dir"] == "e" and kw["refit_terms"] == ("a", "b") and kw["refit_iterations"] == (1, 5)
    assert kw["deep"] == (0, 3) and kw["grad_save"] == (3,) and kw["meta"]["launcher"]["isaaclab_args"] == ["--task", "T"]
    ns.evidence_refit_iterations = ""
    with pytest.raises(SystemExit):
        L._evidence_kwargs(ns, [])
    ns.evidence = None
    assert L._evidence_kwargs(ns, []) is None
