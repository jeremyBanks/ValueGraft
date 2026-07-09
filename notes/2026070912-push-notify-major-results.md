---
name: push-notify-major-results
description: Send a PushNotification when a MAJOR result or milestone lands (not routine progress)
metadata:
  type: feedback
---

Send a **PushNotification** whenever a MAJOR result or milestone lands — proactively, especially when
the user is away (overnight runs). Make it a standing habit, not something to be reminded of.

**What counts as major (notify):** a headline finding (e.g. the fresh-conv reproduction verdict —
does +0.10 hold), per-model results as they land, a key review/verdict (Fable clearing the checkpointing,
an un-anchored trajectory call), an incident/failure, hitting a budget or a decision point that needs them.

**What does NOT (stay quiet):** routine monitor heartbeats, per-conv render progress, "still rendering,"
housekeeping commits. Those go to the monitor stream, not a push.

Keep the notification itself short and substantive — the actual result/number, not "check in." The user
explicitly asked for this (2026-07-09): "send me a notification when you have [results]... make it a thing
you do generally for major results."
