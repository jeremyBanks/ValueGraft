# Primary-source reading: *Models Take Notes at Prefill*

**Author:** Sol (`gpt-5.6-sol-xhigh`)

Source: Bojie Li, [*Models Take Notes at Prefill: KV Cache Can Be Editable
and Composable*](https://arxiv.org/pdf/2606.17107), arXiv:2606.17107v1,
14 June 2026. This is a recent single-author preprint with accompanying code;
the final paper should describe it as prior preprint evidence, not settled consensus.

## What the paper actually demonstrates

The central setup is a prompt containing a policy/rule and a mutable field whose
value controls a later decision. The paper compares a stale cache, refreshing only
the field K/V, recomputing everything downstream of the field, and a clean oracle.
Its causal account is that prefill computes the field-conditioned conclusion and
writes it onto a small set of downstream punctuation/newline/section-boundary
aggregator tokens; the later decision reads those notes rather than re-deriving the
answer from the field.

Primary-source details relevant here:

- field-only refresh has essentially zero decision recovery (reported `−0.028` on
  Llama-3.1-8B and near zero across the other tested families), while recomputing
  the downstream suffix recovers `1.0`;
- the paper summarizes the direct causal share of the field token as under 1%;
- transplanting the eight highest-effect downstream positions recovers roughly
  `0.74–0.79`, while eight random downstream positions recover at most `0.035`;
- holding the field value fixed while flipping one rule token changes the computed
  conclusion; transplanting the downstream notes carries the flipped conclusion
  (`0.998–1.009` recovery), while patching the changed rule token does not;
- the conclusion becomes decodable at downstream aggregators earlier in depth than
  the decision token commits to its answer;
- the mechanism is reported across Qwen3, Llama-3.1, Gemma-2, and Mistral families,
  including controls using conversational phrasing.

The composability result is not a claim that arbitrary stored post-RoPE keys can be
copied bit-for-bit to new positions. The paper explicitly performs RoPE
repositioning and discusses boundary/seam repair; its abstract reports logit cosine
`0.90–0.999` across twelve models. That is useful systems evidence, but it does not
make the rejected native-tail transplant in our frozen layout rotation-free or
confirmatory at our much tighter causal-effect scale.

## Exact relevance to coherent-state v10

The analogy must be mapped correctly. In the prefill paper, the mutable field is
analogous to information in the evicted history, not to the generated summary. In
our correct source, the generated summary is the **latest downstream region after
the entire evicted history, retained tail, summary request, and assistant header**.
It is therefore a plausible carrier of the paper's memoized-note mechanism rather
than merely the original field token whose own K/V the paper found unimportant.

V10 intervenes at the summary boundary and then recomputes assistant-close and
retained-tail rows separately in each arm. Its `G_correct-G_fresh` contrast includes
any effect that correct summary state causally propagates into those later rows.
The preprint therefore does not show that v10 is structurally incapable of detecting
a channel.

It does establish a serious limitation: memoized conclusions may be distributed on
same-position request/header/delimiter tokens outside the summary, or may not be
recapitulated from summary state. Accordingly, a v10 null licenses only:

> No downstream-usable correct-history-specific channel was detected in the
> generated summary rows under this fixed position-preserving assay.

It cannot license “no write-time channel exists elsewhere.” The clean follow-up is
the already-recorded same-position correct-versus-wrong request/header or immediate
post-summary aggregator assay, with matched controls and a separate preregistration.

