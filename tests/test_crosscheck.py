"""Phase 0B Q1 gradient cross-check: Sim2Gate's standalone side against the recorder, on real toy training."""
import json
import shutil

import numpy as np
import pytest

pytest.importorskip("rsl_rl")
from sim2gate.v2 import crosscheck as cc  # noqa: E402
from sim2gate.training.signals import gae as recorder_gae  # noqa: E402
from test_evidence import train  # noqa: E402


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    root = tmp_path_factory.mktemp("bundle")
    ev = root / "runs" / "toy_s3" / "evidence"
    ev.mkdir(parents=True)
    train(3, 0.0, ev, obs_norm=False, sample_envs=16)      # 16 environments: the sample is the whole rollout
    return root


def test_standalone_directions_match_the_recorder(bundle, tmp_path):
    man = cc.reference(str(bundle), str(tmp_path / "ref"))
    assert set(man["files"]) == {"toy_s3_it00000.npz", "toy_s3_it00002.npz"}
    for fn, chk in man["self_checks"].items():
        # on the CPU (no TF32) the rebuilt policy reproduces the recorded one closely, and the neighbour is far off
        assert max(chk["log_prob_max_abs_diff"], chk["mean_max_abs_diff"], chk["std_max_abs_diff"]) < 1e-5
        assert chk["neighbour_mean_max_abs_diff"] > 10 * chk["mean_max_abs_diff"]
        it = fn.split("_it")[1][:5]
        mine = np.load(tmp_path / "ref" / fn)
        rec = np.load(bundle / "runs" / "toy_s3" / "evidence" / "deep" / f"it{it}_directions.npz")
        rn = {str(n): i for i, n in enumerate(rec["names"])}
        names = [str(n) for n in mine["names"]]
        assert names[:2] == ["trainer", "A"] and any(n.startswith("minus:") for n in names)
        for i, n in enumerate(names):
            x, y = mine["directions"][i], rec["directions"][rn[n]].astype(np.float64)
            assert x @ y / (np.linalg.norm(x) * np.linalg.norm(y)) > 0.99999, n


def test_gae_matches_the_recorders(bundle):
    s = np.load(bundle / "runs" / "toy_s3" / "evidence" / "deep" / "it00001_samples.npz")
    meta = json.load(open(bundle / "runs" / "toy_s3" / "evidence" / "meta.json"))
    import torch
    t = {k: torch.from_numpy(np.asarray(s[k], np.float64)) for k in ("raw_rewards", "values", "last_values")}
    want = recorder_gae(t["raw_rewards"], t["values"], t["last_values"], torch.from_numpy(s["dones"] > 0),
                        torch.from_numpy(np.asarray(s["time_outs"], np.float64)), gamma=meta["gamma"],
                        lam=meta["lam"]).numpy()
    got = cc.gae(*(np.asarray(s[k], np.float64) for k in ("raw_rewards", "values", "last_values", "dones",
                                                          "time_outs")), meta["gamma"], meta["lam"])
    assert np.allclose(got, want, atol=1e-9)
    assert np.allclose(got, np.asarray(s["trainer_advantages_unnormalized"], np.float64), atol=1e-4)


def test_compare_passes_on_itself_and_fails_on_a_perturbation(bundle, tmp_path):
    cc.reference(str(bundle), str(tmp_path / "a"))
    shutil.copytree(tmp_path / "a", tmp_path / "b")
    rep = cc.compare(str(tmp_path / "a"), str(tmp_path / "b"))
    assert rep["pass"] and rep["min_cosine"] == pytest.approx(1.0)
    f = tmp_path / "b" / "toy_s3_it00002.npz"
    d = dict(np.load(f))
    d["directions"] = d["directions"].copy()
    d["directions"][0] += 1e-2 * np.linalg.norm(d["directions"][0]) * np.random.default_rng(0).standard_normal(
        d["directions"].shape[1]) / np.sqrt(d["directions"].shape[1])
    np.savez(f, **d)
    rep = cc.compare(str(tmp_path / "a"), str(tmp_path / "b"))
    assert not rep["pass"] and "toy_s3_it00002.npz:trainer" in rep["below_rule"]
    (tmp_path / "b" / "toy_s3_it00000.npz").unlink()
    assert any("different files" in p for p in cc.compare(str(tmp_path / "a"), str(tmp_path / "b"))["problems"])


def test_identity_checks_accept_collection_precision_and_reject_the_wrong_actor(bundle, tmp_path):
    deep = bundle / "runs" / "toy_s3" / "evidence" / "deep"
    meta = json.load(open(bundle / "runs" / "toy_s3" / "evidence" / "meta.json"))
    g, l = meta["gamma"], meta["lam"]
    # GPU-like noise of 1e-3 on the recorded mean (with log pi recomputed from it) is accepted
    s = dict(np.load(deep / "it00002_samples.npz"))
    rng = np.random.default_rng(1)
    s["old_dist_param_0"] = s["old_dist_param_0"] + 1e-3 * rng.standard_normal(s["old_dist_param_0"].shape)
    m, sd, a = (np.asarray(s[k], np.float64) for k in ("old_dist_param_0", "old_dist_param_1", "actions"))
    s["old_log_prob"] = (-((a - m) ** 2) / (2 * sd ** 2) - np.log(sd) - 0.5 * np.log(2 * np.pi)).sum(-1)
    np.savez(tmp_path / "noisy.npz", **s)
    names, *_ , chk = cc.directions(str(deep / "it00002_models.pt"), str(tmp_path / "noisy.npz"), g, l,
                                    str(deep / "it00000_models.pt"))
    assert 5e-4 < chk["mean_max_abs_diff"] < cc.MEAN_TOL
    assert chk["neighbour_separated"]
    # the wrong iteration's actor is rejected (its mean is far off), and so is a changed std
    with pytest.raises(ValueError, match="mean differs"):
        cc.directions(str(deep / "it00000_models.pt"), str(deep / "it00002_samples.npz"), g, l)
    s2 = dict(np.load(deep / "it00002_samples.npz"))
    s2["old_dist_param_1"] = s2["old_dist_param_1"] * 1.01
    np.savez(tmp_path / "std.npz", **s2)
    with pytest.raises(ValueError, match="std differs"):
        cc.directions(str(deep / "it00002_models.pt"), str(tmp_path / "std.npz"), g, l)
