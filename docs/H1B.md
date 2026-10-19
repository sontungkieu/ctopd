# H1b: path versus trunk span supervision

Implementation inputs: Tung's `H1b Implementation Plan Bias vs Variance in Span
Supervision.md` (19 October 2026), and the companion Experiment Design (18 October
2026). The implementation plan takes precedence where the two differ. Baseline
source: `0eeed4ca2b0d7b07dd50076e35d81f030a3b9f2d`.

## Contract

`h1b_span` subclasses ByteAlignCrossTokenizerKD, with five isolated launcher
presets. The shared Strict files, byte codec and loss kernels remain byte-identical
to the baseline; see `H1B_SOURCE.json`. Existing presets keep their defaults.

| Preset | Arm | Detached span coefficient multiplying log q_student_path |
|---|---|---|
| h1b-a | a | zero |
| h1b-b | b | 2 lambda delta_path |
| h1b-b-bl | b_bl | 2 lambda (delta_path - previous_bucket_mean_path) |
| h1b-c-str | c_str | 2 lambda delta_trunk |
| h1b-c | c | 2 lambda (delta_trunk - previous_bucket_mean_trunk) |

Lambda is 0.5. Both strict and span sums use the existing common optimizer-window
token denominator. The baseline is frozen during each complete fit window, then
updated after the optimizer step. Evaluation/no-grad does not update it or window
statistics. State exports contain previous bucket means, arm/lambda and window
index, **not** a complete optimizer/RNG/training resume checkpoint.

The workload is Qwen2.5-7B-Instruct -> Llama-3.2-3B-Instruct, math_code_20k,
512-response rollout/train batches, microbatch 4 with 128 accumulation microsteps,
8,192 generated tokens, 100 updates, and MATH500 mean@32 every 10 updates. H1b
rollouts use top-p 1.0; MATH evaluation independently retains top-p 0.95. There is
one student rank with ring size 1, no final-logit softcap, and no multirank claim.
The optimizer and other flags are inherited unchanged from strict-full.

## Byte groups and probability semantics

Groups use Strict's token-piece bytes and EOS segmentation. Equal complete bytes
are partitioned at shared cumulative byte ends. Content 1:1 groups and paired EOS
feed the original overlap-vocabulary RKL/chunked_loss. Nonempty 1:N/N:1/N:M groups
feed the span term; EOS is excluded. Responses with byte/segment mismatch, empty
pieces or unsupported special IDs are excluded as a whole from both terms and
counted. Offsets and the original denominator include excluded tokens.

**Parity boundary:** arm a has tested loss/all student-gradient parity with Strict
on accepted fixture rows. Strict can still keep a paired EOS when a content segment
has mismatched bytes; H1b's planned whole-response exclusion intentionally removes
that EOS too. Empty-piece/segment rejection is also stricter. Universal equality
on rejected responses is not claimed. Real-response differences are reported by
the alignment validator rather than silently passed.

The trunk probability is the sum of disjoint events: follow the canonical path
for j tokens, then finish or overhang the chunk in one token. It includes the full
canonical path and is a lower bound on the conditional byte-prefix event. All
duplicate byte-equivalent token IDs are retained in covers and exact branches.
Content paths must spell the group bytes exactly. Full-vocabulary head logits are
tiled; only selected path logps retain gradients. Detached cover/overhang masses
are reduced inside checkpointed student tiles; teacher scoring is no-grad.

This does **not** establish unbiased string-KL training. Remaining confounds are:

- Canonical prefix conditioning and outcome-dependent synchronization chunks.
- Re-encoding decoded rollout text instead of preserving the sampled token path;
  this is the implementation plan's choice, shared across arms. H1b logs the raw
  response re-encoding match rate. Top-p 1.0 alone does not eliminate this issue.
- A baseline indexed by the sampled chunk's type/length is outcome-dependent even
  when its means come from the previous window. Previous-window separation removes
  same-sample mean leakage, not all possible score-function bias.
- Whole-response exclusion changes the effective training distribution.

These are ablations of the specified estimators, not proof of the Design's
unbiasedness claims or a complete reproduction of Hou's unreleased span ablation.

## Logging and replay provenance

Every arm computes both targets, both baselines and the same diagnostics. Window
moments weight chunks directly, not averages of microbatch means. JSON summaries
include all 24 buckets, counts and flags for undefined correlations/bias-floor
subsets. Scalar logs use finite zero plus counts/defined flags for empty subsets.
Per-chunk parquet uses null for absent overhang log-mass (log(0)); it never stores
raw prompts, only response hash, chunk hex and diagnostics.

Files live at `checkpoints/h1b/step_K.json` and `chunks_step_K.parquet`. K is the
zero-based pre-update model/window index: 100 updates produce K=0..99. Full dumps
are requested at 0/20/60/100; index 100 is an **offline final-checkpoint scoring**
operation, not a fictitious 101st training window.
The trainer's first training archive is `rollout_data/1.jsonl` (pre-update model
index 0); it does not write a training `rollout_data/0.jsonl`.
Sampling is deterministic from
window/response hash/span and does not advance the training RNG. H1b rollout JSONL
adds the teacher prompt and sampled output IDs so offline tokenization can reuse
the production method. Older archives lack that prompt; tools require explicit
`--assume-shared-prompt` if the caller has independently established equality.

Offline tools require existing local snapshots and `local_files_only=True`; they
never download models or regenerate responses. Reports include archive SHA-256
and model/tokenizer metadata hashes; these metadata hashes do not authenticate
every weight shard. Keep the model/checkpoint provenance alongside each report.

## Commands (company node; existing runtime)

From the committed source root, with the allocated GPU exposed explicitly:

```bash
export PYTHONPATH="$PWD"
RUNTIME=/opt/ctopd-b200-u22
ROOT=/workspace/storage-shared/nlp/tungks/ctopd_all
S="$ROOT/models/Llama-3.2-3B-Instruct"
T="$ROOT/models/Qwen2.5-7B-Instruct"
python3 tools/launch_single_b200.py --algorithm h1b-c --runtime "$RUNTIME" --dry-run
```

If an old Strict archive lacks `tea_prompt`, recover it without regenerating any
responses. This uses the production PromptDataset template builder on the original
parquet pool and demands an exact, unambiguous student-prompt match. It preserves
finish reasons and never fabricates missing sampled IDs:

```bash
bash "$RUNTIME/python-b200.sh" -m tools.h1b.enrich_archive \
  --archive "$OLD_ARCHIVE" --dataset "$ROOT/data/Train/math_code_20k/train.parquet" \
  --student "$S" --teacher "$T" --output "$ROOT/h1b-enriched.jsonl"
ARCHIVE="$ROOT/h1b-enriched.jsonl"
```

The five-step canary retains the original launcher's initial MATH500 evaluation;
previewing a plan consumes no GPU, whereas executing it includes that evaluation.

T1/T3/T9 real-input validation (TEXTS must contain at least 10,000 texts, ARCHIVE
at least 500 real responses, with the production prompt/truncation metadata):

```bash
bash "$RUNTIME/python-b200.sh" -m tools.h1b.validate_alignment \
  --student "$S" --teacher "$T" --texts "$TEXTS" --archive "$ARCHIVE" \
  --output "$ROOT/h1b-alignment.json"
```

M2 trunk table, followed by M3 bounded exhaustive checking on 200 responses:

```bash
bash "$RUNTIME/python-b200.sh" -m tools.h1b.exact_strprob \
  --student "$S" --teacher "$T" --archive "$ARCHIVE" --limit 512 \
  --trunk-only --output "$ROOT/h1b-trunk.jsonl"
bash "$RUNTIME/python-b200.sh" -m tools.h1b.exact_strprob \
  --student "$S" --teacher "$T" --archive "$ARCHIVE" --limit 200 \
  --prune 1e-6 --max-nodes 10000 --output "$ROOT/h1b-exact.jsonl"
```

The checker uses depth-first search with a branch-safe, depth-bounded KV cache
stack. Siblings get independent cache copies, including when an HF cache mutates
in place. `--full-prefix` selects an independent reference path; branches deeper
than `--cache-depth` also fall back to full-prefix forwards. It is not yet batched
across frontier nodes; real-model cache compatibility and B200 throughput/memory
remain unmeasured. The finite search preserves duplicates and records an upper bound on
all pruned/unexpanded probability; only zero omitted mass is called exact.
The conservative gate uses mean *upper-bound* log-probability error < 0.1 mean
absolute path/trunk delta in every populated bucket, for both models. Empty
buckets are `insufficient`, not PASS; a zero gap cannot pass a strict relative
threshold. Both gate policy and raw counts must be inspected before training.
Gate status is currently **NOT MEASURED**. Depth-2 extension is conditional on a
failed measured gate and is not implemented speculatively.

T10 has a separate diagnostic launcher so the standard 100-update presets remain
unchanged. Preview first; `--execute` actually consumes the allocated company GPU:

```bash
python3 -m tools.h1b.canary --algorithm h1b-c --runtime "$RUNTIME" \
  --run-id canary-h1b-c-s42 --root "$ROOT"
# Repeat for each of the five presets, adding --execute only after allocation.
```

Gradient replay uses a fixed 4,096-response archive; it never creates that archive.
Start with 128 responses (two groups of 64) to measure memory/time and sketch error:

```bash
bash "$RUNTIME/python-b200.sh" -m tools.h1b.grad_replay \
  --student "$S" --teacher "$T" --archive "$REPLAY_ARCHIVE" \
  --limit 128 --output "$ROOT/h1b-replay-probe"
bash "$RUNTIME/python-b200.sh" -m tools.h1b.validate_sketch \
  --replay "$ROOT/h1b-replay-probe" --output "$ROOT/h1b-sketch-gate.json"
# After sizing, use --limit 4096, --buckets 0,3 and a fresh output directory.
bash "$RUNTIME/python-b200.sh" -m tools.h1b.fit_cb \
  --replay "$REPLAY_DIR" --bootstrap 200 --output "$ROOT/h1b-curves.json"
```

Replay reuses the algorithm's targets/gradients, with leave-group-out bucket
means. Seven full FP32 buffers are allocated by default (five terms plus two
disjoint strict-position controls), more than the plan's five-buffer estimate.
`--buckets 0,3` adds eight buffers and may exceed B200 memory; `--buffer-device cpu`
is an explicit alternative requiring host RAM and transfers. Exact vectors for
the first two groups can be very large on disk (`--exact-groups 0` disables them,
but then the real T11 gate is not measured). Raw gradients only are implemented;
optional Adam-preconditioned replay and optimizer-state exports are not enabled.
The qualified architecture contract requires zero dropout; checkpointed replay
does not accumulate the training logger/baseline.

Analysis reports c(B), strict-position control cosine, split-half agreement,
finite-pool SNR, norm ratios, group noise scale, fitted c_infinity/B_noise and
group-bootstrap intervals. Token-weighted pooling retains common batch
normalization across differently sized responses. At B=64, two 32-response halves
cannot be recovered from 64-response group sketches: split-half is null, not a
fabricated measurement. At the entire finite pool SNR can be undefined; it is
reported null. Boundary fits are marked not identified. This is not a causal
bias-versus-variance verdict or an automatic decision-table result.

## Validation and milestone status

Run CPU tests only in an existing environment:

```bash
TORCH_COMPILE_DISABLE=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
```

See `H1B_VALIDATION.json` for the executed checks. Fixtures verify algorithmic
identities; they do not replace T1/T3/T9 real-tokenizer/500-response gates, T10
B200 five-preset canaries, or T11 on full model gradient groups.

M1/M2/M4/M5 **code** is implemented with the limitations above. No real trunk
table, M3 gate decision, B200 qualification, gradient curves, trained checkpoint,
new model/data transfer, or M6/M7 scientific result is claimed. No new GPU job was
launched for this implementation. Do not update checkouts used by active jobs;
use fresh source/run IDs and preserve the existing Strict/SimCT/MP runs.
