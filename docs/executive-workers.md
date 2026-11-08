# Executive workers

The experimental worker API proposes changes. It never writes authoritative
state or executes provider tool calls. Existing `LLMAgent` behavior is unchanged.
Tests use fake structural `signal_gating.llm.LLMClient` clients without network
access or credentials.

## Real provider construction

Install the existing optional extra with `pip install 'signal-gating[llm]'`.
The following factory constructs a client but makes no request. Supply the model,
capacity, tokenizer and versioned maximum quote from trusted operator
configuration. No model name, published price or billing guarantee is inferred.

```python
from signal_gating.executive.adapters.openai_worker import OpenAIWorker
from signal_gating.executive.context import TokenCounter
from signal_gating.executive.records import Limits


def provider_worker(
    *,
    model: str,
    capacity: int,
    counter: TokenCounter,
    maximum_quote: Limits,
    quote_version: str,
    quote_source: str,
) -> OpenAIWorker:
    return OpenAIWorker.from_environment(
        api_key_env="OPENAI_API_KEY",  # Environment name, never credential text.
        model=model,
        capacity=capacity,
        counter=counter,
        quote=maximum_quote,
        quote_version=quote_version,
        quote_source=quote_source,
        output_limit_parameter="max_tokens",
        # base_url="https://your-compatible-endpoint.example/v1",
    )
```

`from_environment` imports OpenAI only when called and sets `max_retries=0`.
Injected clients must also disable retries. An exposed nonzero `max_retries` is
rejected; custom client internals remain the operator's responsibility. Scheduling
owns later attempts. The operator owns closing the concrete provider client.
Output parameter choices are `max_tokens`, `max_completion_tokens`, and
`max_output_tokens`. These cannot override model, messages or other controls.

Use the same counter configured for `ContextAssembler`. `UTF8ByteCounter` is a
conservative capacity estimate when no tokenizer is available. Counters must
return actual positive integers. The adapter recounts the canonical rendered
messages, reserves framing plus output against both packet and adapter capacity,
and sends `render_packet(packet)` verbatim. It adds no prompt, schema, tool list
or history. It also rejects locally counted or provider-reported output overruns.
These checks cannot undo a charge already incurred by a nonconforming provider.

## Coordinator contract

`OpenAIWorker.per_call_limits(packet)` exposes trusted per-call maxima. A quote
must cover input plus framing, output, one call and a positive deadline allowance.
For strict monetary admission, it also needs a finite cost maximum and versioned
operator provenance. Set explicit zeros for dimensions known not to be consumed.
A missing bound is unknown, never zero. Quote correctness is an operator/provider
assertion; the SDK is not an external spending sandbox.

`worker.reservation_batch(policy, state, budget_id, episode=episode, packet=packet)`
delegates to `BudgetPolicy.reserve` with `subject_id=episode.id` and the episode's
reservation ID. It returns an event-only `CommitBatch`, including a `worker_quote`
audit event with model, version, source and maximum. It makes no commit or call.
It refuses a mismatched episode/model and a deadline beyond the quote allowance.

The serialized Kernel must merge new subject and reservation post-images into
one `state.changed` event and atomically commit them with the running task,
context packet, private input artifact and quote audit. Union original pre-batch
CAS reads. Commit before inference; release context staging pins in `finally`.
Recheck the policy against current state before reusing a reservation, so ancestor
overages block dispatch. Do not maintain another balance ledger.

Construct `EpisodeRunner(is_current=check_current, max_active=...)`.
`check_current(episode, packet) -> bool` synchronously checks the authoritative
current episode/task revisions and running status, fence, IDs, configured worker,
reservation/admission, freshness and authority. It runs at admission, at actual
child dispatch and after completion. The callback does not commit. Final result
acceptance still needs the coordinator's current checks and CAS. Episode deadlines
must fit both `per_call_limits` and remaining episode/goal time. The optional
`clock` and `wait_until(deadline)` hooks permit controlled deterministic tests;
production defaults use UTC deadlines and event-loop timers on Python 3.10+.

On success, settle `result.usage` through the policy. On `WorkerError`, reconcile
`error.usage`; on cancellation/process loss use `policy.mark_unknown`. Missing or
partial usage remains `status="unknown"`. Only optional provider-envelope
`prompt_tokens` and `completion_tokens` that are nonnegative actual integers are
observed. Model JSON usage is validated as part of the full DTO but then replaced.
Capacity counts and quoted maxima are never billed usage. Calls and locally
measured elapsed time are observations; monetary usage remains unknown unless
separately reconciled with authoritative evidence. Preserve observed overages.

`close()` stops admission, wakes active runners and requests cancellation once;
it returns without waiting for a provider to cooperate. It does not close the
transport or prove the provider stopped. Detached work counts against concurrency
until done. `late_usage_ready` and `take_late_usage()` expose only `(Episode,
Usage)` observations, never stale proposals. The coordinator must serialize their
settlement or uncertainty handling and retain unresolved holds. Undrained
observations also count against concurrency, bounding retained memory. Process
loss can lose those in-memory observations; durable pre-call reservations remain
the recovery authority. Hard termination of malicious same-process work requires
host isolation. Synthetic tests establish these boundary behaviors, not model
quality, production savings or actual provider billing accuracy.
