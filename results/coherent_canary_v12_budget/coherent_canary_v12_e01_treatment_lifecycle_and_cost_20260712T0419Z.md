# V12 e01 diagnostic treatment — lifecycle and immediate cost record

**Recorded by:** Sol — GPT-5.6 Sol, extra-high reasoning

The one authorized diagnostic allocation completed its full artifact-preserving
lifecycle without reaching either timeout.

## Observed execution

- pod: `t09se1chhqpw5e`;
- provider clock began: `2026-07-12T04:01:03Z`;
- exact host: Secure `NVIDIA A100 80GB PCIe`, UUID
  `GPU-b2a992b5-2caf-55c6-51d9-3de66c44468e`, driver `580.159.04`,
  81,920 MiB;
- scientific execution commit:
  `cbdfa481fe08de62bf8178d09ca040716d610f21`;
- raw runner: `PASS`, 868.907 seconds total;
- model preparation: 448.045 seconds;
- treatment computation: 420.750 seconds;
- independent pod harvester: `PASS`;
- primary/placebo records: 31/3; no placebo vector met the frozen
  availability construction, so available-placebo count is 0;
- receipt explicitly records `diagnostic_only=true`,
  `formal_v12_decision_eligible=false`, and
  `aggregate_expansion_authorized=false`.

The intact 8,897,066-byte raw matched receipt SHA-256
`f6622d978c80d8f4f874c208ef9f6d9546ab3f1bf7676655c9a0b4662718d082`.
Its deterministic lossless package compressed to 1,439,866 bytes, reconstructed
byte-for-byte to the same hash, and was committed in result commit
`13928f60924e6fb1377026376963ee65660ac8a2` together with the receipt, pod
harvest, independent local harvest, job log, and cross-host audit.

The precommitted post-run audit returned `PASS`:

- focal treatment-fresh and Phase-A canonical hashes both
  `e742706b58d09f1dd4ebc04da2d1df91a29c637dfd3b81d8e1d96201922df3ff`;
- nonfocal treatment-fresh and Phase-A canonical hashes both
  `3ce032e625e98cf19c125b2fe14e29bddca2a38c9d8cc3402a8526606a20f170`;
- normalized pod/local harvest hashes both
  `cb86d2dd13f20fd6b551ee55534c01747069872945ad07438ca8e0171d1ce1d2`;
- no unexpected harvest difference.

These checks were completed before treatment estimands were inspected.

## Shutdown and immediate cost

After result commit and push, deletion began at `04:19:31.971544Z`; provider
404 was recorded at `04:19:35.120294Z`. A second independent snapshot at
`04:19:47.344776Z` again returned HTTP 404, listed zero active pods, and showed
balance `$61.5865176995`. The detached watchdog process then exited.

The immediately prelaunch balance was `$62.0207254421`, so the conservative
balance delta across this allocation window is `$0.4342077426`, below the
buffered `$0.7963541667` forecast and `$0.90` case ceiling. The whole canary's
conservative origin was `$63.3160022124`; the immediate cumulative delta is
therefore `$1.7294845129`, leaving `$0.6705154871` below the frozen `$2.40`
ceiling. Provider settlement can revise these immediate snapshots; the final
billing-ledger audit remains authoritative.

No pod remains. No further v12 case is authorized.
