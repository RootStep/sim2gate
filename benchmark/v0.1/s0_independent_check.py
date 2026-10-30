"""Independent off-instance check of the S0 run files: validity and reproducibility only (S0 is never analysed).

    python s0_independent_check.py SMOKE_DIR MANIFEST_AT_4a665cc.json MANIFEST_AT_d7e88a6.json
"""
import gzip, hashlib, json, math, sys
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import independent_analysis as ia
import independent_truth as it

D = sys.argv[1]  # directory holding the extracted phase2_tier_r_smoke files
RUNS = {"attempt3": (f"{D}/p0v5b_s101__N080.jsonl.gz", sys.argv[2]),
        "attempt4": (f"{D}/p0v5b_s101__N080.attempt4.jsonl.gz", sys.argv[3])}
cmds = json.load(open(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "configs/tier_r_commands_64.json")))["commands"]
out, recs = {}, {}
for name, (path, man) in RUNS.items():
    m = json.load(open(man))
    h, eps, probs = ia.read_run(path, m, ia._sha(man), "p0v5b_s101", "N080")
    r = {"file_sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(), "read_run_problems": probs,
         "header_manifest_sha_matches": h["manifest_sha256"] == ia._sha(man), "episodes": len(eps)}
    lens, cmd_dev, max_step, zero_moves, min_disp, max_h_jump = set(), 0.0, 0.0, 0, 1e9, 0.0
    for e in eps:
        T = it.validate_episode(e); lens.add(T)
        i = e["env"]
        want = cmds[i % 64][:2]
        cmd_dev = max(cmd_dev, max(abs(c[j] - want[j]) for c in e["cmd_vel_xy"] for j in (0, 1)))
        p, hh = e["base_pos_w_xy"], e["base_height_m"]
        for k in range(1, T):
            d = math.hypot(p[k][0] - p[k-1][0], p[k][1] - p[k-1][1])
            max_step = max(max_step, d)
            if d == 0.0 and hh[k] == hh[k-1]:
                zero_moves += 1
            max_h_jump = max(max_h_jump, abs(hh[k] - hh[k-1]))
        min_disp = min(min_disp, math.hypot(p[-1][0] - p[0][0], p[-1][1] - p[0][1]))
    r.update({"episode_lengths": sorted(lens), "env_ids": sorted(e["env"] for e in eps) == list(range(128)),
              "max_command_deviation_vs_command_set": cmd_dev,
              "max_base_xy_step_m": max_step, "max_base_height_step_m": max_h_jump,
              "identical_consecutive_samples": zero_moves,
              "min_start_to_end_displacement_m": min_disp,
              "header_sync_ok": h["sync_checks"]["ok"], "header_runtime_ok": h["runtime_checks"]["ok"],
              "header_tool_sha256": h["tool_sha256"]})
    out[name] = r
    recs[name] = {e["env"]: e for e in eps}
# reproducibility: same seed, policy, cell and commands
same, maxdiff = 0, 0.0
for i in range(128):
    a, b = recs["attempt3"][i], recs["attempt4"][i]
    if json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True):
        same += 1
    else:
        for k in ("base_height_m",):
            maxdiff = max(maxdiff, max(abs(x - y) for x, y in zip(a[k], b[k])))
        maxdiff = max(maxdiff, max(abs(x[j] - y[j]) for x, y in zip(a["base_pos_w_xy"], b["base_pos_w_xy"]) for j in (0, 1)))
out["reproducibility"] = {"identical_episodes_attempt3_vs_4": same, "max_abs_difference_if_any": maxdiff}
print(json.dumps(out, indent=1))
