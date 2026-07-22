# Focused Improvement Loop

## Goal

SGP must let any model and any harness improve against an explicit capability
target. A trajectory is experience, not learning. The missing layer must turn
evaluation evidence into one focused intervention, compare that intervention
against the incumbent on identical cases, reject regressions, and preserve the
result for the next run.

"FABLE 5 level" is therefore an application-defined objective suite, not a
hard-coded model name. A caller defines the dimensions and target scores that
represent the required capability. SGP supplies the closed-loop control system.

## Contract

The public API consists of:

- `EvaluationCase`: one stable, weighted input.
- `Objective`: one named target, weight, and allowed regression tolerance.
- `Assessment`: per-objective scores plus per-objective evidence.
- `EvaluationSuite`: runs any candidate through caller-provided harness and
  evaluator callables. It supports repeated samples, a fixed-size worker pool,
  deterministic median aggregation, and opt-in raw-output retention. The common
  path retains only evaluator scores and evidence.
- `EvaluationReport`: weighted scores, target progress, and deterministic
  selection of the highest-value weak dimension and its weakest cases.
- `AcceptancePolicy`: accepts a candidate only when total target progress and
  the selected focus improve while aggregate and per-case regressions stay
  within each objective's tolerance.
- `ImprovementLoop`: asks a caller-provided improver for exactly one candidate
  per iteration, evaluates it on the same suite, accepts or rejects it, then
  repeats from the current incumbent.
- `ImprovementHistory`: bounded, digest-verified JSONL records. It stores
  score deltas, focus evidence, per-case candidate evidence, acceptance
  decisions, and regressions, but not arbitrary candidate objects or harness
  outputs.
- `RetentionPolicy`: bounded durable history by age, record count, and exact
  encoded bytes. Defaults are 30 days, 500 records, and 8 MiB. Compaction keeps
  the newest records, atomically rewrites a `0600` file, and rebuilds the digest
  chain from genesis. The active run retains its complete feedback context in
  memory even when durable history compacts.
- `ImprovementEvent`: observable lifecycle signals that can be bridged into a
  mesh without coupling the loop to a particular harness.

Harnesses, evaluators, improvers, candidates, inputs, and outputs are generic.
Callables may be synchronous or asynchronous. Candidate identity is supplied by
the caller so SGP never serializes a model object, credential-bearing client, or
opaque harness configuration.

## Closed loop

1. Evaluate the incumbent on every case and sample.
2. If every objective meets target, stop.
3. Select the objective with the largest weighted target gap. Select its weak
   cases and preserve their evaluator evidence.
4. Give that focused diagnosis and prior experiment history to the improver.
5. Evaluate the candidate on the exact same suite.
6. Accept only if target progress and focused capability improve and no protected
   aggregate or per-case capability regresses.
7. Append the decision to durable history. Repeat from the accepted incumbent or
   from the unchanged incumbent after rejection.

This is hill climbing with explicit guardrails. It does not claim that a model's
weights changed. The candidate can be a prompt, tool policy, routing graph,
memory strategy, model choice, fine-tune identifier, or any compound harness
configuration.

## Acceptance tests

The deterministic test harness must prove:

1. Weighted aggregation and focus selection identify the real weak capability.
2. Multiple noisy samples reduce by median in stable case order.
   Scheduler task count never exceeds `max_concurrency` workers, and a failing
   worker cancels its siblings.
3. Missing scores or evidence fail closed instead of silently skewing results.
4. A focused improver reaches all targets over repeated accepted iterations.
5. A candidate that improves the focus but regresses a protected case is rejected;
   a later safe candidate starts from the unchanged incumbent and is accepted.
6. JSONL history reloads exactly and detects tampering.
7. Lifecycle events expose baseline, focus, candidate, decision, and stop state.
8. Retention removes stale and overflow records, respects the exact byte cap,
   preserves active-run feedback, and reloads as a valid digest chain.

## Limits

- Quality is bounded by the caller's cases and evaluator. SGP cannot make a bad
  judge truthful.
- Default comparison is deterministic, not statistical significance testing.
  Repeated samples reduce noise, while callers set meaningful deltas/tolerances.
- Candidate artifacts remain caller-owned. History records IDs and evidence, not
  executable objects.
- The loop improves inference systems and harness policy. Training model weights
  remains an external candidate-generation mechanism.
