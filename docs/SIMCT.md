# Author-code SimCT in the common CTOPD pipeline

Select `simct` to run the author's released `span_ctkd` objective. This is separate
from Strict and the MP-OPD modes: `mp-atomic`/`mp-fixed1` are not SimCT.

## Source and exact scope

The source is [sunjie279/SimCT-](https://github.com/sunjie279/SimCT-/tree/cf0f33a0e6c967d4b74ea32b2dba12be01b73b9e),
commit `cf0f33a0e6c967d4b74ea32b2dba12be01b73b9e`. The author's released
`scripts/ctopd/qwen25_gemma2_span_mix10k_lr5e-7.sh` selects `span_ctkd` with unit
RKL. Locally the public mode name is `simct`; the donor registration is isolated
and does not register a second legacy alias. The vendor class differs only by
relocating the registration import. Original source fixtures, SHA-256 hashes and
the MIT notice are retained; see [SIMCT_SOURCE.json](SIMCT_SOURCE.json).

The adapter changes the execution interface from the author's full student
logits to CTOPD hidden states plus the student LM head, preserves student final
softcapping, and translates metrics into CTOPD's `train/*` / `distill/*` names.
The author's teacher uses the raw linear head, which is preserved here. It does
not inherit the MP-OPD adapter. Per-response alignment, shared-vocabulary mapping,
span construction, EOS inclusion, unmatched suffix handling, virtual vocabulary
and optional teacher-mass masking all use the pinned author's functions.

The adapter processes one response at a time and checkpoints the student head.
Empty alignment gives a connected zero loss. Nonfinite loss, invalid hidden row
counts and invalid token normalizers fail explicitly. Debug token/logit dumps
are suppressed. Unit RKL, no CE mixing, on-policy KD, one student rank and ring
size 1 are the supported contract. Other configurations fail before launch.

## Important distinction: released code versus paper equations

The [paper's equations 7-8](https://arxiv.org/html/2605.07711v1#S3.SS3) describe
length-normalized continuation **log-probability** scores. The released class
`_build_virtual_vocab_logits` instead averages selected **raw logits** for each
span and appends that value to the raw shared-vocabulary logits. Its comments
claim a geometric-mean probability interpretation, but position-dependent
full-vocabulary normalizers are absent from that computation. These two formulas
are generally different, including under additive logit shifts at later span
positions. This port preserves the released code; it does not silently replace
it with a reconstructed paper objective.

The released alignment is greedy over independently decoded token strings. It
is not the MP-OPD byte-checked atomizer, and may leave an unmatched suffix
unsupervised. These semantics are retained for source parity. This port is not
proof of paper/code equivalence, nor of parity with a different KDFlow revision
whose algorithm is also named `simct`.

## Launch

Inside this checkout on an allocated company node:

```bash
git pull --ff-only
python3 tools/launch_single_b200.py --algorithm simct \
  --runtime /opt/ctopd-b200-u22 --dry-run
```

The standard library dry-run uses no GPU and creates no run. Remove `--dry-run`
only when ready for the real allocated single-B200 workload, with the assigned
GPU exposed through `CUDA_VISIBLE_DEVICES` and all models/data present. Use a new
run ID. No runtime rebuild or new dependency is needed for this source port.

The preset changes only the algorithm and adds `simct_gh_mask_threshold=0`
(disabled, as in the released author launchers). The adapter maps this isolated
flag to the author's optional `span_gh_mask_threshold` semantics. Direct KDFlow
CLI users may set `--simct_gh_mask_threshold` to a positive finite number;
zero disables it. Current launch presets do not enable it.

The common CTOPD recipe retains Qwen-to-Llama, the existing student initialization,
20k prompt source, optimizer batch 512, 100 steps, generation limit 8192, optimizer
and evaluation settings. It does not add SimCT's teacher-generated 10k SFT warmup
or its original model pairs. Compare all methods from the same checkpoint under
this common recipe; label the result a common-pipeline comparison, not a
reproduction of SimCT's published table. The launcher currently evaluates MATH500
only, not the paper's entire benchmark suite.

## Verification

Using an existing compatible CPU PyTorch environment:

```bash
TORCH_COMPILE_DISABLE=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
```

The oracle runs the original saved author class and original RKL numerical bodies,
without importing the adapter or CTOPD's RKL. Only import/decorator wiring is
removed to execute those original bodies in the test process. Compare mixed
spans, strict-only responses, optional G(h) masks, softcapping, loss, every student
gradient, and alignment/mask diagnostics. An independent hand construction checks
virtual logits and RKL gradients. Additional checks cover teacher stop-gradient,
response boundaries, shared microbatch normalization, empty alignment, argument
validation, launcher settings and unchanged existing numerical source hashes.

The [CPU validation receipt](SIMCT_VALIDATION.json) records the 56-test pass and
six original-source comparisons: maximum absolute loss error `0.0`, maximum
absolute student-gradient error `3.725290298461914e-09` in FP32 fixtures.

CPU checks do not qualify target Torch/CUDA, B200/FSDP integration, full-response
memory/throughput or downstream accuracy. Per-response virtual-vocabulary logits
can still be large. No GPU training is launched by the port or tests.
