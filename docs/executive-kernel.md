# Experimental executive kernel

The executive kernel coordinates typed goals, bounded worker episodes, scoped
local actions, evidence verification and recovery in one foreground process.
Its journal is authoritative. Workers propose changes; they do not write kernel
state or grant themselves permissions. This is an experimental API under
`signal_gating.executive`. The compatibility API in `signal_gating.core` is unchanged.

## Install and check the distribution

Use the full tested Git revision in the [README install commands](../README.md#install).
The base install needs Pydantic and PyYAML. The `llm` extra supplies the optional
OpenAI SDK; importing the executive and running the demo do not need it.
Use Python 3.10–3.14. The local single-writer implementation requires POSIX
filesystem locking and descriptor-relative filesystem operations; it fails
closed on unsupported writer platforms. Run in a private directory on a local
filesystem, with one writer per store. Network filesystems are not a supported
locking or durability guarantee.

From any directory after installation:

```bash
signal-gating-executive --help
signal-gating-demo
DEMO_DIR="$(mktemp -d)"
signal-gating-executive demo --store "$DEMO_DIR/release"
signal-gating-executive inspect --store "$DEMO_DIR/release/store" --json
```

`python -m signal_gating.executive` exposes the same operations. The executive
demo uses a deterministic worker, real local workspace adapters and durable
state. It exercises correction during work, stale-proposal rejection, event
coalescing, restricted actions, unknown-effect reconciliation, pause/restart,
retained context and verified completion. Its JSON evaluation must have
`passed: true`, nonempty `summary.completed_goal_ids` and
`metrics.forbidden_dispatches: 0`. It is a synthetic fixture, not evidence of
model quality, production savings or protection against hostile host code.
The static Python 3.10 grammar check in this fixture does not replace the
Python 3.10–3.14 CI matrix.

## Bootstrap an owner and a real local goal

Configuration is closed JSON, not executable plugins. `init --store PATH` only
initializes storage. The first `run` needs `bootstrap_owner: true` and an owner
Principal. Later runs must match the exact persisted identity, including its
creation timestamp. Keep the configuration instead of generating a new owner
on restart. Local CLI access is trusted owner access through the filesystem;
there is no remote authentication service.

The source examples [local-config.json](../examples/executive/local-config.json)
and [goal.json](../examples/executive/goal.json) run a free diagnostic that
deliberately blocks for owner input and produces no task outcome. To prepare an
actual bounded workspace task without a source checkout, run this installed API
example in a new private working directory:

```python
import json
import sys
from pathlib import Path
from signal_gating.executive import (
    ArgvCheck, Criterion, Goal, KernelConfig, Limits, Principal,
    WorkerConfig, WorkspaceConfig,
)

root = Path.cwd().resolve()
workspace = root / "workspace"
workspace.mkdir()
(workspace / "release.py").write_text("def add(a, b):\n    return a + b\n")
owner = Principal(name="Local owner", trust_domain="local",
                  authenticated_origin="local-config", roles=("owner",))
worker = Principal(name="Readiness worker", trust_domain="local",
                   authenticated_origin="configured-worker", roles=("worker",))
check = Criterion(
    description="The release module compiles and its arithmetic assertion passes",
    evaluator_type="deterministic", verifier_id="local-check-v1",
    acceptance_rule="exit_zero", check_id="release-check",
    workspace_inputs=("release.py",),
)
review = Criterion(
    description="Owner accepts the exact readiness report evidence",
    evaluator_type="human", verifier_id="owner-approval-v1",
    acceptance_rule="explicit_owner_approval",
)
goal = Goal(
    owner_id=owner.id, worker_id="analyst",
    outcome=("Inspect release.py. Produce a readiness.md report without changing "
             "release.py. Run release-check for its criterion. Publish report "
             "evidence and request owner review. Do not send or deploy anything."),
    limits=Limits(input_tokens=100000, output_tokens=10000, calls=10,
                  elapsed_seconds=600, cost_microunits=0),
    criteria=(check, review),
)
config = KernelConfig(
    owner=owner, bootstrap_owner=True,
    workers=(WorkerConfig(id="analyst", kind="fixture", principal=worker,
                           context_token_limit=32768, max_output_tokens=2048),),
    workspace=WorkspaceConfig(
        root=workspace, readable_paths=(".",), writable_paths=("readiness.md",),
        named_checks=(ArgvCheck(
            id="release-check", argv=(sys.executable, "-I", "-S", "-c",
                "from pathlib import Path; ns={}; "
                "exec(compile(Path('release.py').read_text(), 'release.py', 'exec'), ns); "
                "assert ns['add'](2, 2) == 4"),
            timeout_seconds=10, max_output_bytes=65536,
            environment={}, maximum_charge_microunits=0,
        ),),
    ),
    verifier_ids=("local-check-v1",),
)
(root / "config.json").write_text(config.model_dump_json(indent=2))
(root / "goal.json").write_text(goal.model_dump_json(indent=2))
print(json.dumps({"goal_id": goal.id, "owner_id": owner.id, "worker_id": worker.id}))
```

The fixture worker in this configuration is intentionally diagnostic. Try it
with a separate diagnostic store, then stop it before configuring real work:

```bash
signal-gating-executive run --store ./diagnostic-state \
  --config ./config.json --goal ./goal.json
```

The service prints JSONL and accepts one tagged command object per stdin line.
Send `{"kind":"inspect"}` for current state, then `{"kind":"stop"}` to close.
EOF also closes cleanly. `--batch` runs until idle: exit 0 means verified selected
goal completion; exit 3 means idle without completion, including a legitimate
owner-input blocker. Exit 2 means startup/configuration rejection; exit 4 means
runtime failure or detected store corruption. Receipt acceptance means a command
was committed, not that its goal succeeded.

## Explicit live model configuration and budgets

Install the README's `llm` extra. Select an actual provider model compatible
with chat completions and the adapter's default `max_tokens` output parameter.
The Python adapter also accepts explicit `output_limit_parameter` and `base_url`;
the JSON CLI configuration does not expose those two options. Models that need
different parameters require explicit Python integration, not inferred options.

Change `config.json` worker `kind` to `openai`, set `model` to that exact model
and `api_key_env` to an environment-variable name such as `SGP_EXECUTIVE_KEY`.
Set that variable privately in the launching shell. Never put its value in
configuration, goals, prompts, commands, evidence or version control.
Supply a `prices` object using reviewed operator/provider bounds:

```text
{
  "version": "your-reviewed-price-bound-version",
  "currency": "USD",
  "input_microunits_per_million_tokens": <your input rate>,
  "output_microunits_per_million_tokens": <your output rate>,
  "maximum_charge_microunits": <your conservative maximum per episode>
}
```

Replace the angle-bracket fields with nonnegative integers. One million USD
microunits is one dollar. This object is trusted configuration, not a model
estimate. The maximum reserves potential cost before each call; rate fields
alone do not establish a charge bound. Set the goal's `cost_microunits` to your
explicit total ceiling and keep all five goal limits finite: input tokens,
output tokens, calls, elapsed seconds and cost. Currency must match. The UTF-8
byte counter is a conservative admission counter, not provider token billing.
Choose capacity/output limits that fit the model and the required context.
The CLI rejects missing `--live`, incomplete budgets or absent maximum quote
before constructing the client or creating the store.

With credentials and a spending budget you have chosen, start a **new** store:

```bash
signal-gating-executive run --store ./readiness-state \
  --config ./config.json --goal ./goal.json --live
```

The adapter disables SDK retries and returns proposals only. The coordinator
commits admission/reservations before model and tool dispatch. Missing actual
usage remains unknown and held; it is never fabricated as zero or silently
replaced by the quote. Local limits cannot guarantee provider invoices or stop
independent calls made elsewhere in the host process. Paid/live inference and
end-to-end model task quality remain unverified in this delivery. A separate
real optional-SDK constructor/close check with dummy credentials and blocked
requests is installation evidence only.

## Grant scoped local commands and review exact effects

No tool permission comes from a model score, a tool name or a claimed role.
The configured worker is bound to a persisted Principal. From the initial
accepted output, use that actual principal ID and the actual goal/owner IDs.
Send a `set_grant` command to the running owner session, for example this
template with all placeholders replaced:

```json
{"kind":"set_grant","grant":{"issuer_id":"OWNER_ID","subject_id":"WORKER_PRINCIPAL_ID","capabilities":["list_files","read_text","write_text","run_check"],"scope":{"owner_id":"OWNER_ID","goal_id":"GOAL_ID","resource_paths":["."]}}}
```

The workspace still restricts writes to `readiness.md`. `run_check` requires
root scope because a trusted configured executable can affect the entire
workspace. For reading/writing alone, use narrower grant paths and omit
`run_check`. Grant updates need the current `expected_revision`; revocation
uses a revised Grant with `revoked: true`. Bootstrap grants apply only when
the owner is first initialized; editing them does not update an existing store.

`list_files` and `read_text` use scoped relative paths. `write_text` binds an
exact prior SHA256 (or `missing`) and an authorized, persisted content Ref.
`run_check` takes a configured command ID, exact criterion Ref and digests
for the full owner-declared `workspace_inputs`. Models cannot supply arbitrary
argv or remove those mandatory inputs. Artifact drafts must first become
persisted artifacts; `draft:key` is not a durable write-content reference.

Checks use exact argv with an absolute executable, without a shell, with a
minimal environment plus explicitly configured additions, timeout and output
bounds. Review the executable and any code it loads. These programs are trusted
host code, not sandboxed workloads. `maximum_charge_microunits: 0` is appropriate
only for a genuinely free local check. Symlinks and escaping paths are refused.

Consequential actions can remain in `pending_reviews`. Copy the **current**
action revision, `arguments_digest`, complete `scope` and `constraint_revision`
from inspection into the owner decision:

```json
{"kind":"decide_action","action_id":"ACTION_ID","expected_revision":1,"decision":"approve","arguments_digest":"EXACT_64_HEX_DIGEST","scope":{"owner_id":"OWNER_ID","goal_id":"GOAL_ID","resource_paths":["."]},"constraint_revision":1,"explanation":"Reviewed these exact arguments and scope"}
```

The numbers above are illustrative. Use inspected values, not guessed
revisions. Change `decision` to `deny` to reject. Approval is bound to those
arguments, current intent, constraints and authority; it does not authorize
changed arguments or later effects. No configured approver means review stays
pending. This distribution provides no outbound email adapter. In Preston's
environment outbound email is human-only and must never be queued or sent by
the kernel, including through a configured program.

## Correct, share, verify and approve evidence

Inspect before every revision-sensitive command. JSON does not accept an
asserted `principal`; the foreground session always uses its trusted owner.
IDs are UUID hex strings. Optional command `id` values and producer/external
identities support exact retry receipts; reuse an identity only for an exact
retry. Changed payload under the same dedup identity is a conflict.

```json
{"kind":"revise_goal","goal_id":"GOAL_ID","expected_revision":1,"outcome":"Inspect release.py and explain compatibility limits in readiness.md","explanation":"Corrected required report scope"}
```

A correction invalidates stale proposals and dependent findings. Hard
constraints require declared rule, verifier or human-review enforcement; prose
alone does not create a proof. Conflicting constraints use `resolve_conflict`
with current goal/conflict revisions and explicit replacements, not a model
vote. Workers cannot remove mandatory dependencies or raise budgets.

Published artifacts start private. To let a worker or the verifier read one,
send the current artifact revision and explicit target principal IDs. Startup
output identifies `worker_principals` and `verifier_principal_id`; the
`principals` inspection list also contains their registered identities:

```json
{"kind":"share_artifact","artifact_id":"ARTIFACT_ID","expected_revision":1,"principal_ids":["VERIFIER_PRINCIPAL_ID","WORKER_PRINCIPAL_ID"]}
{"kind":"attach_evidence","goal_id":"GOAL_ID","criterion_id":"CRITERION_ID","evidence":[{"id":"ARTIFACT_ID","revision":2}]}
```

Sharing advances the artifact revision, so inspect again and use its actual
new Ref. Select only the principals that need those bytes. Inspect the actual
report before owner approval. The trusted local Python API
`ArtifactStore(store).read(ref, principal=owner, goal_id=goal_id)` reads bytes
through current validity/ACL/integrity checks; it also works with
`Store.open_readonly`. Use the exact persisted owner and artifact DTO from
`store.view().records`. CLI inspection exposes metadata, not artifact contents.

A human criterion needs approval of the exact selected evidence and current
goal/criterion revisions. Sharing or an action approval does not imply this:

```json
{"kind":"approve_criterion","goal_id":"GOAL_ID","expected_revision":1,"criterion":{"id":"CRITERION_ID","revision":1},"artifacts":[{"id":"ARTIFACT_ID","revision":2}],"explanation":"I reviewed and accept these exact report bytes"}
{"kind":"reverify","goal_id":"GOAL_ID","criterion_ids":["CRITERION_ID"]}
```

Registered local-check verification authenticates adapter receipts, current
inputs/configuration, ACLs and actual bytes. A forged receipt-shaped artifact
or a worker's claim is insufficient. A `reverify` ACK requests checking; inspect
`findings` for the actual result. A current required criterion, required
commitments and accepted evidence determine completion. Corrections or evidence
invalidation can reopen it. Model judgments, where configured through the
Python API, remain labeled judgments with provenance rather than deterministic
proof.

## Pause, restart, inspect and resolve uncertainty

```json
{"kind":"pause_goal","goal_id":"GOAL_ID","expected_revision":1,"explanation":"Owner pause"}
{"kind":"inspect"}
{"kind":"stop"}
```

Pause is durable. Stop/EOF close the foreground process without inventing
successful effects or zero usage. Inspection can run during the writer session:

```bash
signal-gating-executive inspect --store ./readiness-state --goal GOAL_ID --json
signal-gating-executive run --store ./readiness-state --config ./config.json \
  --resume GOAL_ID --live
```

For fixture-only stores omit `--live`. `--resume` on a paused goal submits the
explicit owner resume at its recovered revision. On an active goal it loads
existing state and emits `status: loaded`, with no fabricated command receipt.
It preserves the existing commitment. Cancelled/failed goals need a new goal.
Restart uses the same private store and exact owner/principal configuration;
do not edit journal/database files to force progress.

`summary.unknown_action_ids`, action status `outcome_unknown`/`needs_human` and
uncertain reservations mean reconciliation is required. An external operation
can succeed before the process records its outcome. The kernel does not claim
exactly-once effects and will not blindly repeat a non-idempotent unknown action.
Use authoritative external/local evidence to resolve the existing action:

```json
{"kind":"resolve_action","action_id":"ACTION_ID","expected_revision":1,"outcome":"succeeded","evidence":[{"id":"EVIDENCE_ID","revision":1}],"explanation":"Verified the original issued operation completed"}
```

Use `failed` only with supporting evidence. Merely resolving an action does not
approve a criterion. For an uncertain reservation, submit `resolve_usage` with
the current reservation revision, known `Usage` and evidence, or explicitly
accept its reserved maximum:

```json
{"kind":"resolve_usage","reservation_id":"RESERVATION_ID","expected_revision":1,"accept_reserved_maximum":true,"explanation":"Owner accepts the recorded maximum for this unresolved charge"}
```

This settles conservatively against the existing bound. It does not establish
the provider's actual charge. Observed overages remain recorded and cannot be
erased by resolution. A reservation without a usable maximum needs evidence of
known usage; never guess a zero to clear a hold.

## Export and subsequent horizons

```bash
signal-gating-executive export --store ./readiness-state --goal GOAL_ID \
  --output ./new-observation.json
```

Export uses read-only current owner inspection, without credentials or a second
writer. The output must be new; exclusive creation uses mode 0600 and refuses
existing files/symlinks. The experimental `goal-observation-v1` projection is
`authority: observation_only`. It retains scoped revisions, causal history,
policy versions, receipts, unknown usage and current authorized artifact
manifests with verified byte integrity. It omits raw prompts, rendered contexts,
artifact bodies, action arguments, machine paths, private/cross-goal references,
credentials and executable grant/binding authority. It is not a backup, replay,
import, remote proof or transferable permission. Integrity labels concern bytes,
not accepted task proof or evaluation benefit claims.

`export_contract_schemas()` exposes 69 complete versioned experimental DTO/union
JSON Schemas independently of optional providers. These describe the full local
types, including fields deliberately absent from portable observations. Schema
validation does not establish cross-field policy, freshness or conformance.
`LegacySignalAdapter` supplies provenance-labeled observation proposals;
`LegacyWorker` wraps an explicitly supplied callable. Both label independent
legacy execution `external_runtime_unmediated`; they do not sandbox Agent/Mesh.

Tasks 1–19 now supply the first local kernel. Adaptive control remains a
subsequent horizon: use recorded events/receipts and explicit evaluations for
offline policy promotion, with no live modification of grants, hard constraints,
verification criteria or ceilings. Federation requires a second implementation
and transport conformance before commitment negotiation. Preserve scoped
authority, explicit acceptance and unknown-outcome semantics across partitions.
See the [delivery status](superpowers/plans/2026-11-07-executive-kernel-status.md)
for validation and independent-review state.
