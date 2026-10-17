"""The blinded-run driver end to end without Isaac Sim: each launch is replaced by writing back a real measurement
(P0 v4 or P0 v2, from docs/mvp-plan) of the checkpoint it was given."""
import json
import os
import random

import pytest

from sim2gate.mvp import acceptance, blinded, tasks
from sim2gate.mvp import planted_bugs as pb
from sim2gate.mvp.baselines import sha256

SOURCES = {"v4s1": lambda: pb._v4(1), "v4s2": lambda: pb._v4(2), "v4s3": lambda: pb._v4(3), "v2s1": lambda: pb._v2(1)}
KEYS = {"check2": "check2_report", "sweep": "sweep_raw", "physx": "physx_raw", "newton": "newton_raw"}
_JOINT = {}


def _fake_joint(tmp_path):
    """A joint sweep from test_conditions' fake robot (its own command set and episode length, so the frozen success
    reference does not apply to it: the report-only summary says so, and the verdict is unaffected)."""
    if "raw" not in _JOINT:
        import test_conditions as tc
        raw, _ = tc._play(tmp_path, mode="joint", name="joint_fake.json")
        _JOINT["raw"] = raw
    return json.loads(json.dumps(_JOINT["raw"]))


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(tasks.BLIND_ALIASES_ENV, raising=False)
    pol = []
    for label, task in (("v4s1", tasks.P0_TASK), ("v4s2", tasks.P0_TASK), ("v4s3", tasks.P0_TASK),
                        ("v2s1", tasks.P0_TASK_V2)):
        ck = tmp_path / "logs" / label / "model_1999.pt"
        ck.parent.mkdir(parents=True)
        ck.write_bytes(label.encode())
        pol.append({"policy": "P0" if label.startswith("v4") else "P2", "seed": int(label[-1]) + (0 if
                    label.startswith("v4") else 10), "checkpoint": str(ck), "task": task})
    calls = []
    (tmp_path / "jointplay").mkdir()

    def fake_run(cmd, log):
        open(log, "w").write("fake\n")
        calls.append(cmd)
        rest = cmd[cmd.index("--") + 1:]
        assert rest[rest.index("--task") + 1].startswith("Sim2Gate-Blind-")
        assert "--external_callback" in rest and os.environ[tasks.BLIND_ALIASES_ENV].endswith("aliases.json")
        ck = rest[rest.index("--checkpoint") + 1]
        out = cmd[cmd.index("--out") + 1]
        m = os.path.basename(out)[:-5]
        if m == "joint":
            assert cmd[cmd.index("--mode") + 1] == "joint" and "physics=isaacsim_physx" in rest
            doc = _fake_joint(tmp_path / "jointplay")
            doc["checkpoint_sha256"] = sha256(ck)
            json.dump(doc, open(out, "w"))
            return 0, ""
        doc = SOURCES[open(ck, "rb").read().decode()]()[KEYS[m]]
        h = sha256(ck)
        alias = rest[rest.index("--task") + 1]                       # what the real recorders record as the task
        if m == "check2":
            doc["checkpoint"] = {"path": ck, "sha256": h}
            doc["play_args"] = rest
        else:
            doc["checkpoint_sha256"] = h
            doc["task"], doc["play_args"] = alias, rest
        json.dump(doc, open(out, "w"))
        return 0, ""

    monkeypatch.setattr(blinded, "run", fake_run)
    return tmp_path, pol, calls


def _prep(tmp, pol):
    return blinded.prepare(pol, str(tmp / "blind"), str(tmp / "key.json"), rng=random.Random(7))


def test_prepare_blinds_ids_and_keeps_the_key_apart(world):
    tmp, pol, _ = world
    man = _prep(tmp, pol)
    assert len(man["policies"]) == 4
    text = (tmp / "blind" / "manifest.json").read_text() + (tmp / "blind" / "aliases.json").read_text()
    for word in ("P0", "P2", "v4s", "v2s", "Velocity", "model_1999"):
        assert word not in text                                   # nothing in the blind folder names a policy
    key = json.load(open(tmp / "key.json"))
    assert {k["policy"] for k in key["policies"].values()} == {"P0", "P2"}
    assert key["manifest_sha256"] == sha256(str(tmp / "blind" / "manifest.json"))
    with pytest.raises(ValueError, match="outside"):
        blinded.prepare(pol, str(tmp / "b2"), str(tmp / "b2" / "key.json"))
    with pytest.raises(ValueError, match="already exists"):
        _prep(tmp, pol)


def test_aliases_resolve_to_the_right_tasks(world):
    tmp, pol, _ = world
    _prep(tmp, pol)
    key = json.load(open(tmp / "key.json"))
    resolved = tasks.blind_aliases(str(tmp / "blind" / "aliases.json"))
    for pid, k in key["policies"].items():
        assert resolved[blinded.alias(pid)] == k["task"]


def test_run_seal_replay_and_score(world):
    tmp, pol, calls = world
    _prep(tmp, pol)
    b, out = str(tmp / "blind"), str(tmp / "out")
    vs = blinded.run_all(b, out)
    assert len(calls) == 20
    key = json.load(open(tmp / "key.json"))["policies"]
    for pid, v in vs.items():
        want = "go" if key[pid]["policy"] == "P0" else "no-go"
        assert v["verdict"] == want, (key[pid], v["checks"])
        assert v["frozen"]["thresholds_sha256"] == "1abb4fcc1b585769c3993b20a22644657efdf391a3eb3f2b054311759e9db739"
    p2 = next(v for pid, v in vs.items() if key[pid]["policy"] == "P2")
    assert "action_latency_ms" in p2["checks"]["1"]["flagged_params"]
    blinded.run_all(b, out)                                         # resume: nothing is measured twice
    assert len(calls) == 20
    for pid, v in vs.items():
        j = v["report_only"]["joint_sweep"]
        assert j["status"] == "invalid" and v["verdict"] in ("go", "no-go")   # report only: never the verdict
        page = open(os.path.join(out, "reports", f"{pid}.html")).read()
        assert pid in page and "Joint sweep (report only" in page and "P0" not in page.replace("P0x", "")
    with pytest.raises(ValueError, match="seal the verdicts"):
        blinded.replay(b, out)
    digest = blinded.seal(out, b)
    assert digest == sha256(os.path.join(out, "verdicts.sha256"))
    with pytest.raises(ValueError, match="sealed"):
        blinded.run_all(b, out)
    pick = blinded.replay(b, out)
    assert len(pick) == 3 and len(calls) == 20 + 15
    for pid in pick:
        a = json.load(open(os.path.join(out, "verdicts", f"{pid}.json")))
        r = json.load(open(os.path.join(out, "replay_verdicts", f"{pid}.json")))
        assert acceptance.signature(a) == acceptance.signature(r)


def test_a_failed_measurement_makes_the_verdict_incomplete_and_blocks_the_seal(world, monkeypatch):
    tmp, pol, calls = world
    _prep(tmp, pol)
    b, out = str(tmp / "blind"), str(tmp / "out")
    real = blinded.run

    def flaky(cmd, log):
        if "newton_mjwarp" in " ".join(cmd):
            open(cmd[cmd.index("--out") + 1] + ".error.txt", "w").write("Traceback\nRuntimeError: boom\n")
            return 1, ""
        return real(cmd, log)

    monkeypatch.setattr(blinded, "run", flaky)
    vs = blinded.run_all(b, out)
    assert all(v["exit_code"] == 2 and "boom" in v["problems"][0] for v in vs.values())
    with pytest.raises(ValueError, match="incomplete"):
        blinded.seal(out, b)
    monkeypatch.setattr(blinded, "run", real)
    vs = blinded.run_all(b, out)                                    # rerun: only the missing runs are launched
    assert all(v["exit_code"] in (0, 1) for v in vs.values())


def test_a_swapped_checkpoint_is_refused(world):
    tmp, pol, calls = world
    _prep(tmp, pol)
    man = json.load(open(tmp / "blind" / "manifest.json"))
    pid = next(iter(man["policies"]))
    (tmp / "blind" / pid / "model.pt").write_bytes(b"v4s2")
    vs = blinded.run_all(str(tmp / "blind"), str(tmp / "out"), only=[pid])
    assert vs[pid]["exit_code"] == 2 and "does not match" in vs[pid]["problems"][0]


def test_frozen_files_match_the_recorded_calibration():
    assert sha256(blinded.FROZEN_THRESHOLDS) == sha256(os.path.join(pb.V4, "thresholds.json"))
    assert sha256(blinded.FROZEN_SUCCESS) == sha256(os.path.join(pb.V4, "checks15", "success.json"))


def test_planted_variants_are_p0_v4_plus_one_change():
    assert sorted(tasks.PLANTED_TASKS) == ["P1", "P2", "P3", "P4", "P5"]
    for pol, task in tasks.PLANTED_TASKS.items():
        v = tasks.VARIANTS[task]
        assert v["base_task"] == tasks.P0_TASK_V4 and v["max_iterations"] == tasks.P0_V4_ITERATIONS
        assert v["experiment_name"] == f"sim2gate_go2_flat_{pol.lower()}_v{v['version']}"
        assert v["version"] == (2 if pol == "P3" else 1)                 # P3-v2 since C9
    assert len({tasks.task_token(t) for t in tasks.VARIANTS}) == len(tasks.VARIANTS)


def test_a_failed_report_only_measurement_does_not_block_the_verdict(world, monkeypatch):
    tmp, pol, calls = world
    _prep(tmp, pol)
    real = blinded.run

    def no_joint(cmd, log):
        if cmd[cmd.index("--mode") + 1:cmd.index("--mode") + 2] == ["joint"] if "--mode" in cmd else False:
            return 1, ""
        return real(cmd, log)

    monkeypatch.setattr(blinded, "run", no_joint)
    vs = blinded.run_all(str(tmp / "blind"), str(tmp / "out"))
    for v in vs.values():
        assert v["exit_code"] in (0, 1) and "joint: no output" in v["report_only"]["problems"][0]
    assert blinded.seal(str(tmp / "out"), str(tmp / "blind"))


def test_p5_ground_and_its_fallback():
    assert tasks.p5_contact() == (2000.0, pytest.approx(2 * 2000 ** 0.5))
    main, fb = tasks.VARIANTS[tasks.PLANTED_TASKS["P5"]], tasks.VARIANTS[tasks.P5_FALLBACK_TASK]
    assert main["changes"][0]["compliant_contact_stiffness"] == 2000.0
    assert main["changes"][0]["fallback"]["task"] == tasks.P5_FALLBACK_TASK
    assert fb["policy"] == "P5" and fb["changes"][0]["compliant_contact_stiffness"] == 3000.0
    assert "fallback" not in fb["changes"][0] and fb["experiment_name"] != main["experiment_name"]
    assert not hasattr(tasks, "P5_PILOT_STIFFNESS_ENV")                 # the pilot setting is gone
