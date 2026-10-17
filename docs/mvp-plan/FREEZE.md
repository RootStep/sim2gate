# Code freeze (MVP plan, step 3): Oct 16, 2026

Approved by VJ (merge go-ahead, Oct 16). Everything the blinded run uses is fixed at this version, before any
planted or held-out policy is trained: the checks, the check 2 thresholds, the episode-success reference, the
command set, the scoring and verdict rules, the A1–A6 readings, test set v1 (P1–P5, with P5's fallback) and the
training, blinding and blinded-run tooling. A change after this point is logged in `CHANGES.md` with its reason;
if it touches anything above, the results say so.

| | |
|---|---|
| Tag / branch | `mvp-freeze-2026-10-16` (branch pinned at the commit; the release tag is added on GitHub by VJ). The commit and tree hashes below are the fingerprint |
| Commit | `7dcece14494aabf09a12adc594802120a42300b3` (main, after merging PRs #1, #2, #3 and #5) |
| Git tree | `4a06c7f78eba6217e970dd64395f84e60699e3b2` |
| `git archive --format=tar --prefix=sim2gate/ mvp-freeze-2026-10-16` SHA-256 (git 2.43) | `1c8cecd02480c177f715619cf053054cd5918b3a844ac22ca0535db99304553d` |
| Frozen plan (unchanged) | `MVP_PLAN_FROZEN_2026-10-12.md`, SHA-256 `9b9d2c85…`; changes C1–C8 in `CHANGES.md` |
| Check 2 thresholds | `src/sim2gate/mvp/frozen_thresholds_p0v4.json`, SHA-256 `1abb4fcc1b585769c3993b20a22644657efdf391a3eb3f2b054311759e9db739` |
| Episode-success reference | `src/sim2gate/mvp/frozen_success_p0v4.json`, SHA-256 `7708a731ded35468a352deb8511da40aa2f987b42dc6d947adfa6c140eef1dee` |
| Command set | `src/sim2gate/mvp/go2_flat_commands_64.json`, SHA-256 `a1d67dd261ba9ab3e2c1fd6abd146cb4307d2df69a31580eb0e9b6a65401f0ec` (command-set id `56222a90…`) |
| Check 1 grid | `65f05de8c3ca4edc1f79e1c4d9aec836b0c2ede7a02d00d0d07db09ccc584146` |
| Joint sweep (report only) | `c43965753041330b88331e65a6beadd4cc361b8af9f4ead8f1cfb0619edfd5a6` |
| Replay draw (A6) | seed 20261015 |

Source files, SHA-256:

```
b08053559e510fee9ffc2ad789a679c0719f8ac52c270f1e477e21004d91dbc4  src/sim2gate/checks/robustness.py
ca9207c3647e593f45de2d9a800f058805924df06008ab9e9f5b310c2270075f  src/sim2gate/checks/transfer.py
cd8686949165d8271225dd924328db6daba3d8a08bd4103e3b6b6d440b56d70e  src/sim2gate/checks/exploits_score.py
24a208d36a0b30701849854c977c6059e112fa3397314b0519fb2084ad35644a  src/sim2gate/mvp/verdict.py
06291c80cc0f05c9178bbd0f3a0c43e71f376bfcb6bc2ed7e36e94fa62df28ad  src/sim2gate/mvp/acceptance.py
871bf6d1d75a8f17395153b4244c7aac01ca73ecd62c11db5bb4d209260053ec  src/sim2gate/mvp/tasks.py
ef6390835ef882b88777481857a09f7e8bf002659054ef7f465feb0ac78d7cdb  src/sim2gate/mvp/blinded.py
ecef523d42f7156cd448eb6bb32085891653a1768055d8d75c490385540aa88f  src/sim2gate/mvp/testset.py
731f641b39a76311552ccfd23b41f4ef6c2b7161e90082bd2e565d203c783483  src/sim2gate/checks/isaaclab_conditions.py
adb2b3d5f1c8b9879a384d1607b2172d426a4b83f9cfdc53f48833d5aa192ead  src/sim2gate/checks/isaaclab_exploits.py
31258f39b78a9ba6337657220acacc4afbed02b990ea6eba5a6fd7774e2d8efd  src/sim2gate/mvp/report.py
```

Check the tree with `git rev-parse 7dcece14494aabf09a12adc594802120a42300b3^{tree}`; every value above can be recomputed from the tag.

Verified before the freeze: 269 tests passing (CI green on `7dcece14494aabf09a12adc594802120a42300b3`); the planted-bug test (all six bugs caught,
control passes); the pre-freeze GPU sessions (`evidence/pre-freeze/`, `evidence/p5-pilot/`, and on Oct 16
evening: `testset check` and `testset smoke` passed inside the container, and a live blinded run of P0 v4 seed 1
with the joint sweep and the HTML report gave **go**).

Next (plan step 4): `python -m sim2gate.mvp.testset train` with Vihaan's public key, then `blinded run`, `seal`,
`replay`; step 5: Vihaan unseals the key and `acceptance` scores A1–A6.

## Vihaan's public key (received Oct 16, 2026)

RSA 3072-bit, `vihaan_public.pem`, SHA-256 `9db064c8f10bd8c4831874064fca7aa7783dac6ca77f97efdfefe2f8cf7f0617`. The test
set's key and training records are sealed to this key only; Vihaan holds the private key.

## After the freeze: change C9 (Oct 16, 2026)

P3 failed the walking check during step 4's training; VJ approved P3-v2 (`CHANGES.md`, C9). Code changed after the
freeze only in `src/sim2gate/mvp/tasks.py` (P3-v2) and `src/sim2gate/mvp/testset.py` (`redefine`, and refusing to
continue or blind while a job waits for a decision). Every other frozen file is unchanged; check with
`git diff 7dcece14494aabf09a12adc594802120a42300b3 -- src/`.

## Known blinding gap found during step 4 (Oct 17, 2026)

`blinded run` launches each measurement through `training.smoke_go2.run`, which streams Isaac Lab's console output to
the screen and to `step4_run.log` (training, by contrast, used `testset.quiet_run`, which writes only to files). Isaac
Lab prints each task's configuration at start-up, including its reward terms, next to the command line that names the
blind ID. Because P3 (no foot-slip penalty) and P4 (no action-rate penalty) differ from P0 in their reward terms, and
P5 in its ground contact, someone reading that output could tie some IDs to their flaw. VJ saw one such reward table
on Oct 17 at about 12:50 (its terms matched P0's, so it does not single out a planted flaw); no ID was read with it
and nothing was changed.

Effect on the result: the verdicts are computed automatically from frozen code, thresholds and success reference, and
are sealed before the key is released, so nobody can act on what the console shows during the run. The protection
that is weakened is the runner's not knowing which ID is which; the remedy for the rest of the run is not to read the
console or the run logs until after sealing. Not fixed mid-run, to keep the frozen code unchanged. For v2: send
`blinded run`'s subprocess output to files only, as `testset` does.
