# Decision models as work-admission gates

Use a small decision model to judge an ambiguous event, then let SGP enforce
the routing policy and trace what happened. The example uses Jev's Noul
probability through the official TypeSafe API. The `Judge` callback can be
replaced by another provider without changing the gate.

## Run without credentials

From a source checkout:

```bash
python -m pip install -e .
python examples/decision_gating.py
```

Five synthetic alerts produce four handler calls, four scripted judgments,
one gate rejection in the SDK trace, and no missed required cases. The
scripted probabilities include uncertainty and a provider failure. This
proves the routing policy, not Jev accuracy or cost savings.

## Run Jev on the same inputs

```bash
export TYPESAFE_API_KEY=...
python examples/decision_gating.py --live --model jev-1.13.0
```

Only this command sends data or incurs inference charges. It sends four
synthetic alert summaries, once each, with no retries. The critical-priority
alert bypasses the model. Each judgment has an outer five-second deadline
and a four-second HTTP timeout. Cancelling the coroutine cannot terminate
an already-running HTTP thread; that call may finish within its own timeout.
The output records the answering model, score, route, and fallback reason.
Exit status is nonzero if a required fixture case is dropped. An all-fallback
run can preserve every case while proving nothing about model capability;
inspect the routes as well as the exit code.

The API shape follows the [TypeSafe quickstart](https://docs.typesafe.ai/introduction/quickstart),
checked October 7, 2026. This integration is not a Chat Completions call.

## Policy and limits

- Priority 7 or higher always reaches the handler without a model call.
- Below that priority, `is_noise >= 0.95` suppresses the handler call.
- Lower scores, invalid answers, HTTP failures, and timeouts preserve work.
- Caller cancellation propagates and does not turn into an admission.
- The 0.95 threshold is illustrative. It has not been calibrated on your data.
- This is a workload filter. Enforce authorization separately in deterministic
  code; a probability cannot grant permission.

Unknown work must not disappear, but low-priority events can still be
misclassified confidently. Typed outputs constrain format, not truth. The
fixture does not establish production incident recall or safe suppression.

## The comparison worth publishing

Freeze a representative, independently labeled alert set before tuning. Split
calibration from holdout evaluation. Compare the ungated handler, deterministic
rules, Jev, and an ordinary structured-output model on the same inputs and
labels. Report:

| Measure | Why it matters |
| --- | --- |
| Missed actionable events, with count and interval | Suppression can hide work |
| Handler calls and model calls | Gating has overhead of its own |
| End-to-end p50/p95 latency | Includes both gate and downstream handler |
| Actual billed cost and cost per correctly handled event | Token estimates alone can mislead |
| Fallback rate and model/build id | Distinguishes reliability from bypass |
| Brier score and reliability bins | Tests whether probabilities match observed outcomes |

Keep every policy fixed on the holdout. Publish the dataset rights, source
commit, environment, raw run records, and failure cases. Claim a benefit only
if it improves workload or cost under the same predeclared quality constraint.
