"""v2 phase 0B analysis (sim2gate.v2.phase0b) on a synthetic bundle made with the real recorder on the toy env."""
import contextlib, io, json, os
import numpy as np
import pytest
import torch

rsl_rl = pytest.importorskip("rsl_rl")
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from sim2gate.training import BSRSPPO  # noqa: E402
from sim2gate.training.evidence import EvidenceRecorder  # noqa: E402
from sim2gate.v2 import phase0b  # noqa: E402
from mountain_car_env import VecMountainCar  # noqa: E402
from test_monitor import cfg  # noqa: E402
from test_evidence import ToyTerms  # noqa: E402

ITERS = 4


def make_bundle(root):
    runs = {n: 0.0 for n in phase0b.PPO_RUNS}
    runs.update(phase0b.BSRS_RUNS)
    for i, (name, eta) in enumerate(runs.items()):
        out = os.path.join(root, "phase0a", "runs", name, "evidence")
        torch.manual_seed(i)
        env = VecMountainCar(num_envs=16, reward="dense", max_episode_length=64, seed=i, init_velocity=True)
        alg = BSRSPPO if eta else "rsl_rl.algorithms:PPO"
        with contextlib.redirect_stdout(io.StringIO()):
            runner = OnPolicyRunner(env, cfg(alg, eta, False), log_dir=None, device="cpu")
            refit = name == phase0b.REFIT_RUN
            rec = EvidenceRecorder(runner, out, terms=ToyTerms(env), deep=range(ITERS), grad_save=range(ITERS),
                                   sample_envs=16, refit_terms=("goal", "effort") if refit else (),
                                   refit_iterations=(1, 2) if refit else ())
            runner.learn(ITERS, init_at_random_ep_len=True)
        rec.close()
    return os.path.join(root, "phase0a")


@pytest.fixture
def toy_rules(monkeypatch):
    monkeypatch.setattr(phase0b, "REFIT_TERMS", ("goal", "effort"))
    monkeypatch.setattr(phase0b, "REFIT_ITERS", (1, 2))
    monkeypatch.setattr(phase0b, "WARMUP", 1)
    monkeypatch.setattr(phase0b, "N_BOOT", 200)


def test_full_analysis_on_synthetic_bundle(tmp_path, toy_rules):
    root = make_bundle(str(tmp_path / "b"))
    out = str(tmp_path / "out")
    assert phase0b.main(["--bundle", root, "--out", out]) == 0
    res = json.load(open(os.path.join(out, "phase0b_results.json")))
    assert res["q1"]["pass"], res["q1"]                     # the recorder's numbers reproduce from the samples
    assert res["q1"]["sample_dumps"] == 10 * ITERS
    assert res["t1_main"]["n_pairs_q0"] == 6 * ITERS * 3
    assert res["t1_main"]["n_pairs_q2"] == 6 * ITERS * 2
    assert len(res["t2"]["points"]) == 4 and res["t2"]["missing"] == 0
    assert len(res["null_refit"]) == 2 and all(-1 <= r["null"] <= 1 for r in res["null_refit"])
    assert res["decision"]["answer"]
    assert "Answer under the frozen rules" in open(os.path.join(out, "phase0b_report.md")).read()


def test_bundle_fingerprint_is_enforced(tmp_path):
    f = tmp_path / "x.tgz"
    f.write_bytes(b"not the bundle")
    with pytest.raises(SystemExit):
        phase0b.main(["--bundle", str(f), "--out", str(tmp_path / "o")])


def test_statistics():
    assert phase0b.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert phase0b.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert phase0b.spearman([1, 1, 2, 3], [1, 2, 3, 4]) == pytest.approx(0.9486833, abs=1e-6)  # ties
    assert phase0b.three_way(-0.05, 0.2, -0.1) == "pass"
    assert phase0b.three_way(-0.5, -0.2, -0.1) == "fail"
    assert phase0b.three_way(-0.2, 0.1, -0.1) == "inconclusive"
