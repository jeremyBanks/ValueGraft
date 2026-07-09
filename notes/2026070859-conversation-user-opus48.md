_This conversation covers the resolution of the wrong-checkpoint saga's
aftermath, a sustained sequence of process corrections about when to consult
Fable versus the user, the discovery and resolution of a nativeness confound
threatening the cross-architecture sweep's validity, and the redesign of the
experimental harness to a per-model-native rendering approach that preserves the
full 16-model ambition._

**Participants:** User and claude-opus-4-8.

**Handoff state.** The native-render verification on Qwen (pod ujg3iqy5cirnga)
was the immediate blocking gate: pending its referent result reproducing ~+0.12,
the plan is to fix QK-norm detection from loaded model modules, finalize the
single pre-registered geometry direction, extend verification to the full
54-scenario set (to test the previously-untested design quality of c13–c54 under
fair native rendering), and then provision and run the full 16-model
per-model-native sweep — factoring in the render-time scaling concern discovered
in this session. No fixed ETA was given for the verification's completion beyond
the revised ~2-hour estimate for the 12-scenario run, which had not yet
completed at the end of this transcript.
