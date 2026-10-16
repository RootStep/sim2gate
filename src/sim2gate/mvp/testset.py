"""Train the test set and blind it, unattended (MVP plan, step 4; VJ's proposal, Oct 16, 2026).

Whoever runs this sees no policy names: training runs in a secret random order with its output written only to
files, and at the end everything labelled (the key, the training logs and settings, the walking-check results) is
encrypted with Vihaan's public key and deleted. What is left is the blind folder (random IDs and checkpoints) for
`python -m sim2gate.mvp.blinded run`, and two sealed files only Vihaan can open.

Vihaan, once, on his own computer (keep the private key there; send only the public key):
    openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:3072 -out vihaan_private.pem
    openssl pkey -in vihaan_private.pem -pubout -out vihaan_public.pem

Inside the Isaac Lab container, from the Isaac Lab-Arena folder:
    python -m sim2gate.mvp.testset check --pubkey vihaan_public.pem --calibration mvp_runs/P0/manifest.json
    python -m sim2gate.mvp.testset train --pubkey vihaan_public.pem --calibration mvp_runs/P0/manifest.json \\
        --work mvp_runs/testset_work --blind mvp_runs/blind --sealed mvp_runs/sealed

Vihaan, after the verdicts are sealed (`blinded seal`):
    python -m sim2gate.mvp.testset unseal --privkey vihaan_private.pem --sealed <folder> --out <folder>

The pool is the plan's 23 policies: P0 seeds 1-3 (the calibration baselines, already trained, from the baselines
manifest), P0 seeds 4-8, and P1-P5 seeds 1-3 (test-set v1 tasks, each P0 v4 plus one change).

Walking check (plan, "Manipulation check"), fixed here before any training: a trained policy walks when at least 80%
of its 64 nominal PhysX episodes succeed (no fall, mean tracking error within the frozen success threshold), the same
measurement and bar as criterion A5. A policy that fails is retrained with seed + 10, then seed + 20; if all three
fail, the run stops for a decision (details sealed). P5 (test-set.md): if any P5 seed fails at stiffness 2000, all
three P5 seeds move to the fallback task (3000) before any retry.
"""
import argparse
import datetime
import json
import os
import secrets
import shutil
import subprocess
import sys
import tarfile

from ..checks import robustness, transfer
from ..training.smoke_go2 import _last_checkpoint, _run_dirs, isaac_python, train_script
from . import blinded, tasks
from .baselines import GO2_COMMANDS, sha256, shutdown_crash

WALK_MIN_SUCCESS = robustness.CLIFF_SUCCESS          # 0.8, as A5
RETRY_SEED_OFFSETS = (10, 20)
CALLBACK = ["--external_callback", "sim2gate.mvp.tasks.register"]
SEALED_DATA, SEALED_KEY = "sealed_for_vihaan.tar.enc", "sealed_for_vihaan.key.enc"


def plan():
    """The jobs to train: (policy, seed, task)."""
    jobs = [("P0", s, tasks.P0_TASK) for s in (4, 5, 6, 7, 8)]
    for pol in ("P1", "P2", "P3", "P4", "P5"):
        jobs += [(pol, s, tasks.PLANTED_TASKS[pol]) for s in (1, 2, 3)]
    return jobs


# ---- quiet subprocess (nothing labelled on the console) ------------------------------------------------------------

def quiet_run(cmd, log_path):
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    return proc.returncode, open(log_path).read()


# ---- encryption (OpenSSL: RSA-OAEP for a random AES-256 key) ------------------------------------------------------

# Isaac Sim's Python points programs at its own bundled libraries (LD_LIBRARY_PATH and the like); the system's openssl
# fails when it loads those instead of its own. Run it with them removed.
_OPENSSL_ENV_DROP = ("LD_LIBRARY_PATH", "LD_PRELOAD", "OPENSSL_CONF", "OPENSSL_MODULES", "OPENSSL_ENGINES")


class OpenSSLError(ValueError):
    pass


def _openssl(*args, **kw):
    env = {k: v for k, v in os.environ.items() if k not in _OPENSSL_ENV_DROP}
    r = subprocess.run(["openssl", *args], capture_output=True, env=env, **kw)
    if r.returncode != 0:
        err = (r.stderr or b"").decode(errors="replace").strip().splitlines()
        raise OpenSSLError(f"openssl {args[0]} failed (exit {r.returncode}): {' | '.join(err[-3:]) or 'no message'}")
    return r


def check_pubkey(pubkey):
    if shutil.which("openssl") is None:
        raise ValueError("openssl is not installed here; sealing needs it")
    out = _openssl("pkey", "-pubin", "-in", pubkey, "-noout", "-text").stdout.decode()
    if "Public-Key" not in out:
        raise ValueError(f"{pubkey} is not a public key")
    probe = os.path.join(os.path.dirname(os.path.abspath(pubkey)), ".sim2gate_probe")
    try:
        with open(probe, "wb") as f:
            f.write(b"probe")
        _openssl("pkeyutl", "-encrypt", "-pubin", "-inkey", pubkey, "-pkeyopt", "rsa_padding_mode:oaep",
                 "-pkeyopt", "rsa_oaep_md:sha256", "-in", probe, "-out", probe + ".enc")
    finally:
        for f in (probe, probe + ".enc"):
            if os.path.exists(f):
                os.remove(f)
    return out.splitlines()[0].strip()


def seal(paths, pubkey, sealed_dir, arc_root):
    """Encrypt `paths` (files or folders, stored relative to arc_root) for the holder of pubkey's private key."""
    os.makedirs(sealed_dir, exist_ok=True)
    tar_path = os.path.join(sealed_dir, ".sealing.tar")
    key_path = os.path.join(sealed_dir, ".sealing.key")
    try:
        with tarfile.open(tar_path, "w") as t:
            for p in paths:
                t.add(p, arcname=os.path.relpath(p, arc_root))
        with open(key_path, "wb") as f:
            f.write(secrets.token_bytes(32).hex().encode())
        _openssl("enc", "-aes-256-cbc", "-pbkdf2", "-iter", "200000", "-salt", "-pass", f"file:{key_path}",
                 "-in", tar_path, "-out", os.path.join(sealed_dir, SEALED_DATA))
        _openssl("pkeyutl", "-encrypt", "-pubin", "-inkey", pubkey, "-pkeyopt", "rsa_padding_mode:oaep",
                 "-pkeyopt", "rsa_oaep_md:sha256", "-in", key_path, "-out", os.path.join(sealed_dir, SEALED_KEY))
    finally:
        for f in (tar_path, key_path):
            if os.path.exists(f):
                os.remove(f)
    return {n: sha256(os.path.join(sealed_dir, n)) for n in (SEALED_DATA, SEALED_KEY)}


def unseal(privkey, sealed_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    key_path = os.path.join(out_dir, ".unsealing.key")
    tar_path = os.path.join(out_dir, ".unsealing.tar")
    try:
        _openssl("pkeyutl", "-decrypt", "-inkey", privkey, "-pkeyopt", "rsa_padding_mode:oaep",
                 "-pkeyopt", "rsa_oaep_md:sha256", "-in", os.path.join(sealed_dir, SEALED_KEY), "-out", key_path)
        _openssl("enc", "-d", "-aes-256-cbc", "-pbkdf2", "-iter", "200000", "-pass", f"file:{key_path}",
                 "-in", os.path.join(sealed_dir, SEALED_DATA), "-out", tar_path)
        with tarfile.open(tar_path) as t:
            t.extractall(out_dir, filter="data")
    finally:
        for f in (key_path, tar_path):
            if os.path.exists(f):
                os.remove(f)
    return out_dir


# ---- training and the walking check -------------------------------------------------------------------------------

def train_command(task, seed):
    return isaac_python() + [train_script(), "--rl_library", "rsl_rl", "--task", task, "--seed", str(seed)] \
        + CALLBACK + [transfer.PHYSX]


def walk_command(task, ckpt, out):
    return isaac_python() + ["-m", "sim2gate.checks.isaaclab_conditions", "--mode", "nominal", "--commands",
                             GO2_COMMANDS, "--out", out, "--", "--task", task, "--checkpoint", ckpt] \
        + CALLBACK + [transfer.PHYSX] + list(transfer.FIXED_OVERRIDES) + list(transfer.ORDERING)


def walking(raw, success):
    rate = robustness.success_rate(raw["episodes"], success["threshold_mps"])
    return {"success_rate": rate, "walks": rate is not None and rate >= WALK_MIN_SUCCESS,
            "falls": sum(bool(e["fell"]) for e in raw["episodes"]), "episodes": len(raw["episodes"]),
            "checkpoint_sha256": raw.get("checkpoint_sha256")}


def _save(doc, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=1)
    os.replace(tmp, path)


def _say(msg):
    print(f"[sim2gate] {msg}", flush=True)


def _train_one(job, work, success, runner):
    """Train job's current (task, seed) and run its walking check. Updates job in place."""
    pol, seed, task = job["policy"], job["seed"], job["task"]
    tag = f"{pol}_{task.rsplit('-', 1)[-1]}_s{seed}"
    before = _run_dirs()
    rc, text = runner(train_command(task, seed), os.path.join(work, "logs", f"train_{tag}.log"))
    new = sorted(_run_dirs() - before)
    ck = _last_checkpoint(new[0]) if len(new) == 1 else None
    att = {"task": task, "seed": seed, "train_returncode": rc, "run_dir": os.path.abspath(new[0]) if new else None}
    crash = shutdown_crash(rc, text, ck, (tasks.VARIANTS.get(task) or {}).get("max_iterations")) if ck else None
    if crash:
        att["train_shutdown_crash"] = crash
    if ck is None or (rc != 0 and not crash):
        att["error"] = f"training exited {rc}; {len(new)} new run folder(s)"
        att["walks"] = False
        job["attempts"].append(att)
        return False
    att["checkpoint"] = {"path": os.path.abspath(ck), "sha256": sha256(ck)}
    out = os.path.join(work, "walk", f"walk_{tag}.json")
    if os.path.exists(out):
        os.remove(out)
    runner(walk_command(task, ck, out), os.path.join(work, "logs", f"walk_{tag}.log"))
    if not os.path.exists(out):
        att.update(walks=False, error="the walking check wrote no result")
    else:
        w = walking(json.load(open(out)), success)
        att.update(walk=w, walks=w["walks"] and w["checkpoint_sha256"] == att["checkpoint"]["sha256"])
    job["attempts"].append(att)
    return att["walks"]


def train_all(pubkey, calibration, work, blind, sealed, runner=quiet_run, rng=None):
    rng = rng or secrets.SystemRandom()
    check_pubkey(pubkey)
    cal = _calibration(calibration)
    success = json.load(open(blinded.FROZEN_SUCCESS))
    for d in (work, os.path.join(work, "logs"), os.path.join(work, "walk")):
        os.makedirs(d, exist_ok=True)
    spath = os.path.join(work, "state.json")
    if os.path.exists(spath):
        state = json.load(open(spath))
    else:
        jobs = [{"policy": p, "seed": s, "base_seed": s, "task": t, "attempts": [], "status": "pending"}
                for p, s, t in plan()]
        rng.shuffle(jobs)
        state = {"version": 1, "started": _now(), "jobs": jobs, "p5_fallback": False}
        _save(state, spath)
    n = len(state["jobs"])
    while True:
        pending = [j for j in state["jobs"] if j["status"] == "pending"]
        if not pending:
            break
        job = pending[0]
        k = state["jobs"].index(job) + 1
        _say(f"job {k:02d}/{n}: training (output in the work folder)")
        ok = _train_one(job, work, success, runner)
        if ok:
            job["status"] = "walks"
            _say(f"job {k:02d}/{n}: walking check passed")
        elif job["policy"] == "P5" and not state["p5_fallback"]:
            state["p5_fallback"] = True                 # test-set.md: all P5 seeds move to the fallback ground
            for j in state["jobs"]:
                if j["policy"] == "P5":
                    j.update(task=tasks.P5_FALLBACK_TASK, seed=j["base_seed"], status="pending")
            _say(f"job {k:02d}/{n}: walking check failed; a predeclared rule changes some jobs (sealed)")
        else:
            tried = len([a for a in job["attempts"] if a["task"] == job["task"]])
            if tried <= len(RETRY_SEED_OFFSETS):
                job["seed"] = job["base_seed"] + RETRY_SEED_OFFSETS[tried - 1]
                _say(f"job {k:02d}/{n}: walking check failed; retraining with a new seed")
            else:
                job["status"] = "failed"
                _save(state, spath)
                _say(f"job {k:02d}/{n}: failed the walking check three times. Stopping for a decision; the details "
                     f"are in the work folder, for Vihaan only.")
                return 1
        _save(state, spath)
    return finish(state, cal, pubkey, work, blind, sealed)


def _calibration(path):
    m = json.load(open(path))
    out = []
    for s in ("1", "2", "3"):
        ck = ((m.get("seeds") or {}).get(s) or {}).get("checkpoint") or {}
        if not ck.get("path") or not os.path.exists(ck["path"]) or sha256(ck["path"]) != ck.get("sha256"):
            raise ValueError(f"calibration baseline P0 seed {s}: checkpoint missing or changed ({path})")
        out.append({"policy": "P0", "seed": int(s), "checkpoint": ck["path"], "task": m.get("task", tasks.P0_TASK)})
    if out[0]["task"] != tasks.P0_TASK:
        raise ValueError(f"the calibration baselines were trained on {out[0]['task']}, not {tasks.P0_TASK}")
    return out


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def finish(state, cal, pubkey, work, blind, sealed):
    """Blind all 23, seal everything labelled for Vihaan, delete the plaintext."""
    pols = list(cal)
    for j in state["jobs"]:
        att = j["attempts"][-1]
        pols.append({"policy": j["policy"], "seed": j["base_seed"], "task": att["task"],
                     "checkpoint": att["checkpoint"]["path"]})
    _save({"policies": pols}, os.path.join(work, "private.json"))
    key_path = os.path.join(work, "key.json")
    blinded.prepare(pols, blind, key_path)
    # the labelled training records: settings, logs and final checkpoints (not every intermediate checkpoint)
    kept = os.path.join(work, "runs")
    run_dirs = sorted({a["run_dir"] for j in state["jobs"] for a in j["attempts"] if a.get("run_dir")})
    for rd in run_dirs:
        dst = os.path.join(kept, os.path.basename(os.path.dirname(rd)), os.path.basename(rd))
        last = _last_checkpoint(rd)
        shutil.copytree(rd, dst, ignore=lambda d, names: [n for n in names if n.startswith("model_")
                                                           and os.path.join(d, n) != last])
    state["finished"] = _now()
    _save(state, os.path.join(work, "state.json"))
    hashes = seal([work], pubkey, sealed, os.path.dirname(os.path.abspath(work)))
    for rd in run_dirs:                                  # the labelled training folders leave the instance's open disk
        shutil.rmtree(rd, ignore_errors=True)
    shutil.rmtree(work)
    _save({"version": 1, "sealed": _now(), "files": hashes, "blind_manifest_sha256":
           sha256(os.path.join(blind, "manifest.json")), "policies": len(pols)},
          os.path.join(sealed, "sealed_manifest.json"))
    _say(f"{len(pols)} policies blinded into {blind}. Key and training records sealed for Vihaan in {sealed}:")
    for name, h in hashes.items():
        _say(f"  {name}  sha256 {h}")
    _say("Nothing labelled remains outside the sealed files. Next: python -m sim2gate.mvp.blinded run")
    return 0


def smoke(pubkey, work, runner=quiet_run, iterations=20):
    """Pre-freeze GPU test of this pipeline on one throwaway policy (P2's task, seed 99, a few rounds; it will not
    walk, which is fine): quiet training, checkpoint discovery, the walking check and sealing all run for real; the
    run folder and the sealed files are deleted afterwards. Prints what it saw; names are fine, nothing is blinded."""
    check_pubkey(pubkey)
    success = json.load(open(blinded.FROZEN_SUCCESS))
    for d in (work, os.path.join(work, "logs"), os.path.join(work, "walk")):
        os.makedirs(d, exist_ok=True)
    task, seed = tasks.PLANTED_TASKS["P2"], 99
    before = _run_dirs()
    rc, text = runner(train_command(task, seed) + ["--max_iterations", str(iterations)],
                      os.path.join(work, "logs", "train.log"))
    new = sorted(_run_dirs() - before)
    ck = _last_checkpoint(new[0]) if len(new) == 1 else None
    _say(f"training exit {rc}; new run folders {len(new)}; checkpoint {ck}")
    if ck is None:
        return 1
    out = os.path.join(work, "walk", "walk.json")
    rc2, _ = runner(walk_command(task, ck, out), os.path.join(work, "logs", "walk.log"))
    if not os.path.exists(out):
        _say(f"walking check wrote no result (exit {rc2})")
        return 1
    w = walking(json.load(open(out)), success)
    _say(f"walking check (exit {rc2}): success {w['success_rate']}, falls {w['falls']}/{w['episodes']}, "
         f"same checkpoint {w['checkpoint_sha256'] == sha256(ck)}")
    sealed = os.path.join(work, "sealed")
    hashes = seal([os.path.join(work, "logs"), os.path.join(work, "walk")], pubkey, sealed, work)
    sizes = {n: os.path.getsize(os.path.join(sealed, n)) for n in hashes}
    _say(f"sealed: {sizes}")
    for rd in new:
        shutil.rmtree(rd, ignore_errors=True)
    shutil.rmtree(work)
    _say("SMOKE OK" if all(v > 0 for v in sizes.values()) else "SMOKE FAILED")
    return 0 if all(v > 0 for v in sizes.values()) else 1


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m sim2gate.mvp.testset", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("check", "train"):
        q = sub.add_parser(name)
        q.add_argument("--pubkey", required=True)
        q.add_argument("--calibration", required=True, help="the P0 baselines manifest (P0 seeds 1-3)")
        if name == "train":
            q.add_argument("--work", required=True)
            q.add_argument("--blind", required=True)
            q.add_argument("--sealed", required=True)
    sm = sub.add_parser("smoke", help="pre-freeze GPU test on one throwaway policy")
    sm.add_argument("--pubkey", required=True)
    sm.add_argument("--work", default="mvp_runs/testset_smoke")
    u = sub.add_parser("unseal")
    u.add_argument("--privkey", required=True)
    u.add_argument("--sealed", required=True)
    u.add_argument("--out", required=True)
    a = p.parse_args(argv)
    try:
        if a.cmd == "check":
            _say(f"public key OK: {check_pubkey(a.pubkey)}")
            _say(f"calibration baselines OK: {len(_calibration(a.calibration))} checkpoints")
            _say(f"Isaac Lab train script: {train_script()}")
            _say(f"{len(plan())} policies to train; frozen success reference {blinded.FROZEN_SUCCESS}")
            return 0
        if a.cmd == "train":
            return train_all(a.pubkey, a.calibration, a.work, a.blind, a.sealed)
        if a.cmd == "smoke":
            return smoke(a.pubkey, a.work)
        _say(f"unsealed into {unseal(a.privkey, a.sealed, a.out)}")
        return 0
    except (ValueError, OSError) as e:
        print(f"[sim2gate] {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
