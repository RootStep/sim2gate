"""deploy/brev/verify_gpu_evidence.py: the GPU evidence carries its own verdict."""
import importlib.util
import json
import os

import pytest

HERE = os.path.dirname(__file__)
spec = importlib.util.spec_from_file_location(
    "verify_gpu_evidence", os.path.join(HERE, "..", "deploy", "brev", "verify_gpu_evidence.py"))
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)

SHA = "979a49e" + "0" * 33
CK = "f7ad0353165b5c36f8b213b1c8c83452652f767aa11ed4f2c0245c444a5a5246"


def _folder(tmp_path, commit=SHA, pkg=SHA, status="passed", samples=4, ck=CK, method=True, other_ok=True,
            update="passed"):
    (tmp_path / "sim2gate-versions.txt").write_text(f"sim2gate_commit={commit}\nsim2gate 0.1.0.dev0 {pkg}\n")
    (tmp_path / "sim2gate_smoke_identity").mkdir()
    ident = {"name": "eta 0 with Sim2Gate attached is bit-identical to stock (final checkpoints)",
             "ok": status == "passed", "status": status,
             "detail": {"stock_vs_repeat": {"differing": 0 if status != "inconclusive" else 3},
                        "stock_vs_eta0": {"differing": 0 if status == "passed" else 1}, "checkpoints": {}}}
    other = {"name": "trainer residual", "ok": other_ok, "detail": ""}
    (tmp_path / "sim2gate_smoke_identity" / "report.json").write_text(json.dumps({"checks": [other, ident]}))
    cs = {"samples_per_control_step": samples, "physics_substeps_per_control_step": 4,
          "covered_share": samples / 4, "physics_handles_decimation": samples == 1,
          "scene_updates_per_step_histogram": {str(samples): 1000}}
    if method:
        cs["method"] = "max of the sensor's current force over every scene update in the control step"
    if update is not None:
        (tmp_path / "sim2gate_update_identity.json").write_text(json.dumps(
            {"status": update, "stock_vs_stock": {"differing": 0}, "stock_vs_sim2gate": {"differing": 0},
             "monitor_records_written": 1}))
    (tmp_path / "sim2gate_check2_eta1_v2.json").write_text(json.dumps(
        {"control_steps": 1000, "checkpoint": {"sha256": ck}, "contact_force_substeps": cs, "metrics": {}}))
    return tmp_path


def _outside(tmp_path, open_ports=(), code=0, probed=3, with_code=True):
    lines = ["# exposure probe from pc to 3.142.199.227"]
    lines += [f"tcp {p} {'OPEN' if p in open_ports else 'closed-or-filtered'}" for p in range(80, 80 + probed)]
    lines.append(f"# {len(open_ports)} of {probed} probed TCP ports reachable")
    if with_code:
        lines.append(f"exit code {code}")
    p = tmp_path / "exposure_outside.txt"
    p.write_text("\n".join(lines) + "\n")
    return str(p)


def _sshd(tmp_path, password="no", kbd="no", pubkey="yes"):
    p = tmp_path / "sshd_auth.txt"
    p.write_text(f"pubkeyauthentication {pubkey}\npasswordauthentication {password}\n"
                 f"kbdinteractiveauthentication {kbd}\npermitrootlogin without-password\n")
    return str(p)


def _verify(folder, outside=None, sshd=None):
    argv = [str(folder), "--commit", SHA, "--checkpoint-sha256", CK] + (["--outside", outside] if outside else [])
    argv += ["--sshd", sshd] if sshd else []
    rc = v.main(argv)
    return rc, json.load(open(folder / "VERIFY.json"))


def test_all_good_passes(tmp_path):
    f = _folder(tmp_path)
    rc, out = _verify(f, _outside(tmp_path))
    assert rc == 0 and out["overall"] == "passed"
    assert all(i["status"] == "passed" for i in out["items"].values())


@pytest.mark.parametrize("kw,item", [
    ({"commit": "0fa715a" + "0" * 33}, "pinned commit"),
    ({"pkg": "0fa715a" + "0" * 33}, "pinned commit"),            # host updated, package not reinstalled
    ({"samples": 1}, "contact force 4/4"),                       # engine-handled decimation: 1 of 4
    ({"samples": 3}, "contact force 4/4"),
    ({"method": False}, "contact force 4/4"),                    # report from the old (history) method
    ({"ck": "0" * 64}, "contact force 4/4"),                     # wrong policy measured
    ({"status": "failed"}, "eta 0 identity, final checkpoints"),
    ({"update": "failed"}, "eta 0 identity, one update"),
    ({"other_ok": False}, "smoke test"),
])
def test_each_requirement_can_fail(tmp_path, kw, item):
    rc, out = _verify(_folder(tmp_path, **kw), _outside(tmp_path))
    assert rc == 1 and out["overall"] == "failed" and out["items"][item]["status"] == "failed"


def test_irreproducible_stock_is_inconclusive_not_passed(tmp_path):
    rc, out = _verify(_folder(tmp_path, status="inconclusive"), _outside(tmp_path))
    assert rc == 3 and out["items"]["eta 0 identity, final checkpoints"]["status"] == "inconclusive"
    assert out["items"]["eta 0 identity, one update"]["status"] == "passed"     # reported separately, not merged


def test_missing_update_identity_is_inconclusive(tmp_path):
    rc, out = _verify(_folder(tmp_path, update=None), _outside(tmp_path))
    assert rc == 3 and out["items"]["eta 0 identity, one update"]["status"] == "inconclusive"


@pytest.mark.parametrize("kw", [{"open_ports": (81,), "code": 1}, {"with_code": False}, {"code": 1},
                                {"probed": 0}])
def test_outside_probe_must_show_no_open_port_and_exit_0(tmp_path, kw):
    rc, out = _verify(_folder(tmp_path), _outside(tmp_path, **kw))
    assert rc == 1 and out["items"]["ingress from outside"]["status"] == "failed"


def test_missing_outside_probe_is_inconclusive(tmp_path):
    rc, out = _verify(_folder(tmp_path))
    assert rc == 3 and out["items"]["ingress from outside"]["status"] == "inconclusive"


def test_short_commit_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        v.main([str(_folder(tmp_path)), "--commit", "979a49e", "--checkpoint-sha256", CK])


def _probe_text(tmp_path, results):
    lines = [f"tcp {p} {'OPEN' if o else 'closed-or-filtered'}" for p, o in results]
    n = sum(o for _, o in results)
    lines += [f"# {n} of {len(results)} probed TCP ports reachable", f"exit code {1 if n else 0}"]
    p = tmp_path / "exposure_outside.txt"
    p.write_text("\n".join(lines) + "\n")
    return str(p)


BREV_PORTS = [(22, True), (53, False), (80, False), (5173, False), (8086, False)]


def test_open_ssh_passes_only_with_key_only_sshd(tmp_path):
    out = _probe_text(tmp_path, BREV_PORTS)
    rc, doc = _verify(_folder(tmp_path), out, _sshd(tmp_path))
    item = doc["items"]["ingress from outside"]
    assert rc == 0 and item["status"] == "passed" and "22" in item["detail"]["allowed_exceptions"][0]


@pytest.mark.parametrize("kw", [{}, {"password": "yes"}, {"kbd": "yes"}, {"pubkey": "no"}])
def test_open_ssh_without_key_only_evidence_fails(tmp_path, kw):
    out = _probe_text(tmp_path, BREV_PORTS)
    sshd = _sshd(tmp_path, **kw) if kw else None
    rc, doc = _verify(_folder(tmp_path), out, sshd)
    item = doc["items"]["ingress from outside"]
    assert rc == 1 and item["status"] == "failed" and "22" in item["detail"]["unexpected_open_ports"]


def test_any_other_open_port_fails_even_with_key_only_ssh(tmp_path):
    out = _probe_text(tmp_path, [(22, True), (80, True), (5173, False)])
    rc, doc = _verify(_folder(tmp_path), out, _sshd(tmp_path))
    assert rc == 1 and doc["items"]["ingress from outside"]["detail"]["unexpected_open_ports"] == ["80"]
