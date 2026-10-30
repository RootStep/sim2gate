"""Dry run of run_stage.py against a fake `docker` (no Isaac Lab): preflight, retries, quarantine, cap stop."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

HERE = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((HERE / "G2_TIER_R_MANIFEST.json").read_text())

FAKE = textwrap.dedent('''\
    #!/usr/bin/env python3
    # fake docker: exec -w CROOT CONTAINER cmd...   (maps CROOT to HOST_ROOT)
    import hashlib, json, os, sys
    a = sys.argv[1:]
    assert a[0] == "exec" and a[1] == "-w"
    croot, cmd = a[2], a[4:]
    host = os.environ["HOST_ROOT"]
    m = lambda p: p.replace(croot, host, 1) if p.startswith(croot) else os.path.join(host, p)
    if cmd[0] == "sha256sum":
        p = m(cmd[1])
        if not os.path.exists(p):
            print("sha256sum: no such file", file=sys.stderr); sys.exit(1)
        print(hashlib.sha256(open(p, "rb").read()).hexdigest(), cmd[1]); sys.exit(0)
    assert cmd[0] == "timeout"
    args = cmd[3:]
    get = lambda k: args[args.index(k) + 1]
    pid, cid, out = get("--policy"), get("--cell"), m(get("--out"))
    state = os.path.join(host, "fake_state.json")
    st = json.load(open(state)) if os.path.exists(state) else {}
    key = pid + "__" + cid
    st[key] = st.get(key, 0) + 1
    json.dump(st, open(state, "w"))
    mode = json.loads(os.environ.get("FAKE_MODES", "{}")).get(key, "ok")
    if mode == "always_fail" or (mode == "fail_first" and st[key] == 1):
        sys.exit(1)
    sys.path.insert(0, os.environ["TEST_DIR"]); sys.path.insert(0, os.environ["TOOLS"])
    import test_independent_pipeline as T, independent_analysis as ia
    man = os.path.join(os.environ["TOOLS"], "G2_TIER_R_MANIFEST.json")
    mm = json.load(open(man))
    hc = {"stage": get("--stage")}
    if mode == "sync_fail_first" and st[key] == 1:
        hc["sync_checks"] = {"ok": False, "problems": ["x"]}
    T.write_run(out, mm, ia._sha(man), pid, cid, 4, T=20, header_changes=hc)
    sys.exit(3 if hc.get("sync_checks") else 0)
    ''')


class RunStageDryRun(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.host = Path(self.tmp.name) / "host"
        self.tools = self.host / "v2_runs/phase2_eval_tools"
        (self.tools / "configs").mkdir(parents=True)
        for f in ("run_stage.py", "independent_analysis.py", "independent_truth.py"):
            shutil.copy(HERE / f, self.tools / f)
        m = copy.deepcopy(MANIFEST)
        m["episodes_per_cell_repeat"], m["command_set"]["n_commands"] = 1, 4
        st = next(s for s in m["stages"] if s["id"] == "S1")
        st["status"] = "APPROVED"
        st["runs"] = [["p0v5b_s101", "N080"], ["s301", "F040"], ["s306", "F045"], ["s311", "A1"]]
        self.m = m
        # fake checkpoints at the manifest paths, with the manifest's digests replaced by the fake files' digests
        for pid, v in m["policies"].items():
            p = self.host / v["checkpoint"]
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(pid.encode())
            v["checkpoint_sha256"] = hashlib.sha256(pid.encode()).hexdigest()
        (self.tools / "G2_TIER_R_MANIFEST.json").write_text(json.dumps(m))
        fake = Path(self.tmp.name) / "docker"
        fake.write_text(FAKE)
        fake.chmod(0o755)
        self.env = dict(os.environ, HOST_ROOT=str(self.host), TOOLS=str(self.tools), TEST_DIR=str(HERE / "tests"))
        self.fake = str(fake)

    def tearDown(self):
        self.tmp.cleanup()

    def drive(self, *args, modes=None):
        env = dict(self.env, FAKE_MODES=json.dumps(modes or {}))
        return subprocess.run([sys.executable, str(self.tools / "run_stage.py"), "--host-root", str(self.host),
                               "--container-root", "/workspaces/isaaclab_arena", "--docker", self.fake] + list(args),
                              env=env, capture_output=True, text=True)

    def out_dir(self):
        return self.host / next(s for s in self.m["stages"] if s["id"] == "S1")["output_dir"]

    def test_preflight_match_and_mismatch(self):
        r = self.drive("--stage", "S1", "--preflight")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("19/19 checkpoints MATCH", r.stdout)
        p = self.host / self.m["policies"]["s306"]["checkpoint"]
        p.write_bytes(b"tampered")
        r = self.drive("--stage", "S1", "--preflight")
        self.assertEqual(r.returncode, 1)
        self.assertIn("MISMATCH s306", r.stdout)
        p.unlink()
        r = self.drive("--stage", "S1", "--preflight")
        self.assertIn("MISSING  s306", r.stdout)

    def test_run_retry_quarantine_and_resume(self):
        r = self.drive("--stage", "S1", "--run", modes={"s301__F040": "fail_first", "s306__F045": "sync_fail_first",
                                                         "s311__A1": "always_fail"})
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)               # s311 A1 missing after two attempts
        summ = json.loads((self.out_dir() / "stage_S1_summary.json").read_text())
        self.assertEqual(summ["counts"], {"ok": 3, "invalid": 0, "missing": 1}, summ)
        self.assertTrue(any(x.name.startswith("s306__F045.jsonl.gz.") for x in (self.out_dir() / "invalid").iterdir()))
        log = [json.loads(x) for x in (self.out_dir() / "stage_S1.jsonl").read_text().splitlines()]
        att = {(x["policy"], x["cell"]): x["attempt"] for x in log if "attempt" in x}
        self.assertEqual(att[("p0v5b_s101", "N080")], 1)
        self.assertEqual(att[("s301", "F040")], 2)
        self.assertEqual(att[("s311", "A1")], 2)
        # re-invocation: valid outputs are skipped, only the missing run is attempted again
        r = self.drive("--stage", "S1", "--run")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(r.stdout.count("attempt 1"), 1)
        self.assertIn("ok 4, invalid 0, missing 0", r.stdout)

    def test_cap_stops_before_run(self):
        st = next(s for s in self.m["stages"] if s["id"] == "S1")
        st["gpu_hour_cap"] = 0.02                                          # 72 s < one run's 103.5 s estimate
        (self.tools / "G2_TIER_R_MANIFEST.json").write_text(json.dumps(self.m))
        r = self.drive("--stage", "S1", "--run")
        self.assertIn("STOPPED before p0v5b_s101 N080", r.stdout)
        self.assertNotIn("attempt 1", r.stdout)

    def test_unapproved_stage_refused(self):
        st = next(s for s in self.m["stages"] if s["id"] == "S1")
        st["status"] = "NOT_APPROVED"
        (self.tools / "G2_TIER_R_MANIFEST.json").write_text(json.dumps(self.m))
        r = self.drive("--stage", "S1", "--run")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("not APPROVED", r.stderr)


if __name__ == "__main__":
    unittest.main()
