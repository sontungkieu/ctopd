# TRUST-R/B/P and SimCT paper-score modes

These opt-in source modes retain the existing CTOPD data/model/optimizer/evaluation
pipeline. They add no SFT warmup and do not reproduce either paper's full recipe.
Existing Strict, SimCT author-code, Atomic/Fixed/Random, GRASS and ALIGN files stay
unchanged. Shared actor branches activate only for the new TRUST-B/P modes.

| Launcher algorithm | Registered mode | Loss intervention |
| --- | --- | --- |
| `mp-trust-r` | `mp_opd_trust_r` | One lambda per response: strict + lambda*mismatch |
| `mp-trust-b` | `mp_opd_trust_b` | One lambda for the complete optimizer batch |
| `mp-trust-p` | `mp_opd_trust_p` | One strict-side projection scale for the complete optimizer batch; mismatch kept |
| `simct` | `simct` | Unchanged released author-code raw-logit span scores |
| `simct_paper` | `simct_paper` | Full-native-vocabulary log-probability scores before virtual-vocabulary normalization |

## TRUST provenance and scope

The numerical kernels are copied byte for byte from
`sontungkieu/SimCT@529ee4b691732e1040d6a45f05fd12f58def96ec`:
`_mp_opd_trust.py`, `_mp_opd_trust_batch.py`, `_mp_opd_trust_p.py`. Existing frozen
head-pair and Gram dependencies are reused. [TRUST_PAPER_SOURCE.json](TRUST_PAPER_SOURCE.json)
records original fixtures and unchanged existing algorithm hashes. The donor MIT
notice remains in the vendored package.

Strict atoms are exactly `one_to_one` from byte-synchronized atomization; mismatch
atoms are `multi_token`. Unknown types fail. There is no extra `xtoken` chunking
for TRUST. Credits, detached rates, summed atom NLL, EOS masking and common token
normalizer are those of the existing MP-OPD adapter.
Here "strict loss" is the atomic path-credit surrogate on `one_to_one` atoms;
it is not the enumerated shared-vocabulary KL of the separate Strict presets.

Let S and M be the summed output-head gradients of strict/mismatch atom losses.
TRUST-R and B use `lambda=max(0,<S,M>/||M||^2)` with donor rules for empty or
degenerate sides; strict loss keeps coefficient 1. R uses one response, B the
entire optimizer window. P uses
`strict_scale=1-min(0,<S,M>)/||S||^2`, keeping mismatch coefficient 1; its donor
rounding protects the halfspace. With no usable strict anchor, P retains Atomic
and flags the waived guarantee. These are head-output-edge proxies, including
bias/softcap, not full-Transformer or Adam descent guarantees.

### Complete-window calibration

B/P cannot silently calibrate each microbatch. A first train-mode no-grad pass
streams two full-head accumulators across every response, including all
cross-response terms. Parameters stay fixed. The second pass restores each
microbatch's Python/NumPy/Torch/CUDA RNG before backward, applies one detached
calibration to the whole optimizer window and advances logical RNG only once.
Only the normal final accumulated step updates parameters. Calibration state is
cleared afterward, including on failures; no graphs or batch logits are stashed.

One student rank/ring size 1, a complete optimizer window starting at step 0,
and one finite positive token normalizer are required. Partial windows, missing
preparation, changed normalizers and invalid geometry fail explicitly. No legacy
microbatch TRUST-B alias is provided. The donor's historical error-to-Atomic
fallback is not reproduced: invalid geometry/memory checks fail with the cause.
This difference affects failure handling, not the valid-input loss parity claim.

Evaluation loss uses its own evaluation-set calibration, unit sample weights,
no backward/optimizer update and no training-calibration reuse. The original
student train/eval mode is restored. MATH500 generation evaluation is unchanged.

TRUST requires unit RKL weight/temperature and exact-head geometry.
`mp_opd_trust_eps_g=1e-12`; `mp_opd_trust_vocab_chunk=4096`. B/P require two
FP32 [vocabulary, hidden] accumulators (plus bias when present), so full-size
memory and throughput still need GPU qualification. No runtime rebuild is needed.

## What `simct_paper` means

This mode corrects the score calculation to
[SimCT v1 equations 7-8](https://arxiv.org/html/2605.07711v1#S3.SS3): first compute
`log_softmax` over the full native vocabulary at every position; use the first
position's log-probabilities for shared candidates and mean selected
log-probabilities for each realized span; then normalize over the virtual candidate
set and calculate RKL. Both sides use the same score convention. Gradients flow
through the full-vocabulary student normalization; teacher scores are detached.

`simct` remains untouched. `simct_paper` is a paper-score reconstruction on the
released author's candidate/alignment interface, **not a claim that all paper
semantics are implemented**. It retains greedy decoded-token alignment, the
released realized-span candidate set, EOS/shared-vocabulary rules, optional G(h)
mask (default disabled), raw teacher-head convention, common normalizer and the
CTOPD initialization/data. It does not add arbitrary alternative-continuation
forward scoring at every prefix, fix all alignment limitations, restore SFT
warmup, or establish equivalence to Rethinking's unavailable `simct` implementation.
Those are separate scientific changes, not hidden inside a score correction.

Tests independently construct Eq.7-8 candidate scores and RKL, compare complete
adapter loss/all student gradients, and verify invariance to independent additive
logit shifts at each position. Matching the released raw-logit loss is expected
only in special cases; numerical parity is asserted against the equations here.

## Commands and verification

From the company checkout, print each full plan without launching a GPU job:

```bash
git pull --ff-only
python3 tools/launch_single_b200.py --algorithm mp-trust-r --runtime /opt/ctopd-b200-u22 --dry-run
python3 tools/launch_single_b200.py --algorithm mp-trust-b --runtime /opt/ctopd-b200-u22 --dry-run
python3 tools/launch_single_b200.py --algorithm mp-trust-p --runtime /opt/ctopd-b200-u22 --dry-run
python3 tools/launch_single_b200.py --algorithm simct_paper --runtime /opt/ctopd-b200-u22 --dry-run
```

Only remove `--dry-run` with the actual allocated GPU, ready models/data and a new
run ID. The real batch512/generation8192/100-step workload remains unchanged.
Use an existing CPU Torch environment for tests; no dependency installation:

```bash
TORCH_COMPILE_DISABLE=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
```

The TRUST oracle separately loads original donor kernels and the TRUST-R method
body; frozen atom/Gram/credit helpers are source-verified shared dependencies.
Tests cover full loss/all student gradients, complete-batch cross-response terms,
projection/no-anchor behavior, dropout RNG replay, a real optimizer update through
the actor integration helper, evaluation state isolation and unchanged older
modes. CPU fixtures do not qualify B200/FSDP, full-size memory, training resume,
accuracy or paper-score reproduction. No GPU job is launched by this port.

The [CPU receipt](TRUST_PAPER_VALIDATION.json) records 76 passing tests and six
TRUST donor comparisons: maximum absolute loss error `0.0`, maximum absolute
student-gradient error `5.960464477539063e-08`. Paper-score checks compare against
independent formulas, not against the author's different raw-logit objective.
