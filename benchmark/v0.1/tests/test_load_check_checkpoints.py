"""Synthetic tests for load_check_checkpoints.py with a stand-in torch module (no real checkpoints, no torch)."""
import contextlib
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _T:
    def __init__(self, shape, vals):
        self.shape, self.vals, self.dtype = tuple(shape), vals, "torch.float32"

    def is_floating_point(self):
        return True

    def isfinite(self):
        ok = all(math.isfinite(v) for v in self.vals)
        return types.SimpleNamespace(all=lambda: ok)


def _decode(o):
    if isinstance(o, dict) and "__t__" in o:
        return _T(o["shape"], [float(v) for v in o["vals"]])
    return {k: _decode(v) for k, v in o.items()} if isinstance(o, dict) else o


def _load(path, map_location=None, weights_only=True):
    d = json.loads(Path(path).read_text())
    if d.get("needs_pickle") and weights_only:
        raise RuntimeError("unsupported global")
    return _decode(d["obj"])


FAKE_TORCH = types.SimpleNamespace(is_tensor=lambda o: isinstance(o, _T), load=_load)


class LoadCheck(unittest.TestCase):
    def setUp(self):
        self._saved = sys.modules.get("torch")
        sys.modules["torch"] = FAKE_TORCH
        import load_check_checkpoints as lc
        self.lc = lc
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.register = self.d / "reg.csv"
        self.rows = []

    def tearDown(self):
        if self._saved is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = self._saved
        self.tmp.cleanup()

    def ckpt(self, name, width=3, value=0.5, nan=False, pickle=False, register=True):
        t = lambda s: {"__t__": 1, "shape": s, "vals": ["nan" if nan else value] * math.prod(s)}
        obj = {"iter": 1999, "model_state_dict": {"actor.0.weight": t([4, width]), "actor.0.bias": t([4])},
               "optimizer_state_dict": {"state": {"0": {"exp_avg": t([4, width])}}}}
        p = self.d / f"{name}.pt"
        p.write_text(json.dumps({"obj": obj, "needs_pickle": pickle}))
        if register:
            self.rows.append({"policy_id": name, "checkpoint_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                              "hash_status": "BYTE_VERIFIED"})
        return p

    def run_tool(self, files, expect):
        with self.register.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["policy_id", "checkpoint_sha256", "hash_status"])
            w.writeheader()
            w.writerows(self.rows)
        out, js = io.StringIO(), self.d / "out.json"
        with contextlib.redirect_stdout(out):
            rc = self.lc.main([str(p) for p in files] + ["--register", str(self.register), "--expect", *expect,
                                                          "--json", str(js)])
        return rc, out.getvalue(), json.loads(js.read_text())

    def test_consistent_files_pass(self):
        rc, out, rec = self.run_tool([self.ckpt("a", value=0.1), self.ckpt("b", value=0.2)], ["a", "b"])
        self.assertEqual(rc, 0)
        self.assertTrue(rec["all_pass"])
        self.assertEqual({f["parameters"] for f in rec["files"]}, {16})  # optimizer tensors excluded
        self.assertEqual({f["iter"] for f in rec["files"]}, {1999})

    def test_structure_mismatch_fails(self):
        rc, out, rec = self.run_tool([self.ckpt("a", value=0.1), self.ckpt("c", width=5)], ["a", "c"])
        self.assertEqual(rc, 1)
        self.assertIn("structure differs", out)

    def test_nonfinite_fails_and_fallback_recorded(self):
        rc, out, rec = self.run_tool([self.ckpt("d", nan=True, pickle=True)], ["d"])
        self.assertEqual(rc, 1)
        f = rec["files"][0]
        self.assertFalse(f["finite"])
        self.assertTrue(f["load_mode"].startswith("weights_only failed"))

    def test_missing_and_unregistered_fail(self):
        rc, out, rec = self.run_tool([self.ckpt("a"), self.ckpt("x", value=0.9, register=False)], ["a", "zz"])
        self.assertEqual(rc, 1)
        self.assertEqual(rec["missing"], ["zz"])
        self.assertIn("digest not registered under an expected ID", out)

    def test_unloadable_file_fails(self):
        bad = self.d / "bad.pt"
        bad.write_text("not a checkpoint")
        self.rows.append({"policy_id": "bad", "checkpoint_sha256": hashlib.sha256(bad.read_bytes()).hexdigest(),
                          "hash_status": "BYTE_VERIFIED"})
        rc, out, rec = self.run_tool([bad], ["bad"])
        self.assertEqual(rc, 1)
        self.assertFalse(rec["files"][0]["loaded"])


if __name__ == "__main__":
    unittest.main()
