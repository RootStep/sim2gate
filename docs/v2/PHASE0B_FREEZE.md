# v2 phase 0B: freeze record

| | |
|---|---|
| Frozen file | `docs/v2/PHASE0B_PROTOCOL.md` |
| SHA-256 | `239124e256207f942782175c0525c4367cfb1ed2d1557e8b0f5d4d70521e7330` |
| Decided by | VJ, Oct 18, 2026 (T1 margin 0.1; T2 PPO-noise yardstick; offline null refit added; all deep iterations, warm-up-excluded check reported) |
| Analyses run on phase 0A data before this commit | None. Only the outcome measurements (slip, jitter, walking) in `evidence/phase0a/README.md` had been looked at; the protocol does not test them. |

Check: `sha256sum docs/v2/PHASE0B_PROTOCOL.md` must print the hash above. Any later change to the protocol is an
amendment, listed below with its date, reason and VJ's approval, and must come before the update-level quantities
are computed.

## Amendments

- Oct 18, 2026, minutes after the freeze and before any analysis: a cross-reference in 4.1 corrected ("see 4.3" to "see 4.5"); the hash above is the corrected file's. First frozen hash was 500f1c9ec8ceb0cade371fbbb75ac66dff5897bc16d7069724226110fda982a3.
