# Gates 1–6 evidence index — Diagnostic Benchmark v0.1

**Recorded:** 2026-10-27 UTC  
**Status:** Archived and SHA-256 verified by operator; Gate 6 **behavioral checks passed, overall report status PARTIAL**. **Not a benchmark freeze or training authorization.**

## Canonical raw snapshot (Brev; not uploaded to GitHub)

- Isolated checkout: `/home/ubuntu/rootstep-benchmark`
- Archive: `/home/ubuntu/rootstep-benchmark/benchmark/v0.1/evidence/gates-1-6-20261027T184824Z`
- Archive-time Git HEAD: `7ca80e405557f1f1ab2680d3e52259cc199f56e8`
- SHA-256 of `MANIFEST.json`: `1d6f59ae1c596428d1a506d6c8d23015419566679fc8bcb35e6eec55c500e97b`
- `SHA256SUMS` verification: **13/13 OK**, reported by operator on 2026-10-27.
- Contents: nine Gate 6 console logs; Gate 3 semantics JSON; three Gate 6 JSON reports. The archive script's inventory had zero missing files. **This is not an exhaustive inventory of all Gate 1–5 evidence.**
- Raw artifact hashes: consult the archive's `SHA256SUMS`; individual hashes were not supplied in chat and are intentionally not invented here.

## Gate 6 machine-readable evidence

| Report | Source commit recorded | Checks | Failures | Original status/stage |
|---|---|---:|---:|---|
| `rootstep-gate6b-ramp-report.json` | `9d4865a` | 28 | 0 | `PARTIAL / before_cleanup` |
| `rootstep-gate6c-varying-report.json` | `92f598a` | 38 | 0 | `PARTIAL / before_cleanup` |
| `rootstep-gate6d-fixed-report.json` | `7ca80e4` | 39 | 0 | `PARTIAL / before_cleanup` |

Total: **105/105 reported checks true**, with no reported errors. The three reports overlap in checks and are **not 105 independent experiments**.

The corresponding logs show `H0_closed` and `D2_closed`, followed by `before_cleanup`; no `finished` checkpoint is present. Environment closure is observed, but completion of Isaac Sim `app.close()` is not proven.

## Supported technical findings

- Actual articulated Go2 instances were created for H0 and D2 with explicit PhysX preset resolution and real `DelayedDCMotor` / `DelayBuffer` instances.
- H0 delay ramp was sampled using **controlled motor counter assignments**, not 76,800 naturally executed simulation steps; D2 remained zero delay.
- Bounded changing-action live steps produced stale delayed position commands for H0 (forced lag 2), while D2 (lag 0) remained immediate. Motor effort and joint-state traces were recorded; no standalone causal torque-effect estimate is claimed.
- Two-environment selective `motor.reset([0])` preserved environment 1's delay-buffer lag/history for all three channels. This did **not** exercise the complete environment reset-manager pathway.
- No PPO training, optimizer, checkpoint generation, benchmark scoring, or four-arm pilot was performed.

## Gate 1–5 status and evidence gap

Earlier operator-reported gate results: Gate 1 **7/7**, Gate 2 **4/4**, Gate 3 **10/10**, Gate 4 **18/18**, Gate 5 **165/165**. Only Gate 3's JSON was explicitly included in this 13-file snapshot. Locate and separately archive the original Gate 1, 2, 4 and 5 artifacts and their source revisions before claiming a complete six-gate reproducibility package.

## Provenance and next steps

The archive-time HEAD (`7ca80e4`) is historical and must not be silently replaced by later branch commits. Subsequent documentation/reporting-source corrections include `d7779a3` (shutdown checkpoint instrumentation) and `3108e45` (corrected qualifications); **neither retroactively changes the archived results**.

1. Copy the archive to protected storage outside the Brev instance; verify checksums at destination. SHA-256 alone does not make files immutable.
2. Recover and hash missing Gate 1, 2, 4 and 5 raw evidence.
3. Review pilot manifestation criteria, independent scoring rules, and stop conditions; freeze before new policy training.
4. Obtain **separate approval** for GPU profiling and the H0×1/T1×1/D1×1/D2×1 pilot.

**No training, GPU profiling, or benchmark freeze is authorized by this evidence index.**
