# ALIGN-NEW from SimCT

The SimCT inventory at `529ee4b691732e1040d6a45f05fd12f58def96ec` contains
`align` (v1) and `align_new`. No implementations explicitly named ALIGN v2/v3
were found in its current branches, algorithm/docs files or relevant commit
messages; the user confirmed no additional version pointer. This port preserves
the real donor name, without assigning a new version number.

## Modes

`mp-align` remains ALIGN v1: cyclic coefficient-space PCGrad within each chunk.
The separate `mp-align-new` first constructs the same local candidate, then
checks the **summed output-head update across the entire response**, including
cross-chunk terms. It interpolates toward Atomic only as much as required to
retain at least `rho=0.1` of Atomic's first-order descent in this proxy geometry.
This is the donor's specified guard, not an exception fallback or a norm cap.
A vanished anchor deliberately yields Atomic and sets the no-anchor metric.

The geometry is the LM-head output edge (weight plus bias if present), not the
full Transformer, tied-input embedding path or Adam-preconditioned update. Positive
proxy descent is not an accuracy guarantee. The guard uses detached coefficients;
no gradient flows through its beta or geometry. Precision context is restored.

New code lives in `kdflow/algorithms/mp_opd_align_new.py` and two additional donor
kernels inside `_mp_opd_port/`. The existing `mp_opd.py`, ALIGN v1, GRASS-Chunk,
Strict and author-code SimCT numerical files remain unchanged. Shared edits add
the mode, isolated arguments and launcher choice. Donor MIT notice is retained
in the same vendored directory. [Source hashes](ALIGN_NEW_SOURCE.json) pin the
new files, original test fixtures and unchanged dependencies.

## Run configuration

Print the full plan from this checkout on the company node:

```bash
git pull --ff-only
python3 tools/launch_single_b200.py --algorithm mp-align-new \
  --runtime /opt/ctopd-b200-u22 --dry-run
```

Default chunks come from the donor tokenizer/string aligner (`xtoken`), with no
learned vocabulary projection. The explicit fixed-run ablation is
`--mp-chunk-source run --mp-chunk-run-length 2`. Both sources are tested. Generation,
optimizer batch, training iterations, models/data and MATH500 evaluation remain
the common CTOPD workload. Runtime archive and dependencies do not change.

The separate KDFlow arguments are:

- `mp_opd_align_new_min_descent_fraction=0.1` (finite, between 0 and 1).
- `mp_opd_align_new_eps_g=1e-12` (finite, nonnegative).

ALIGN-NEW requires `exact_head`, unit KD weight/temperature, RKL, one student rank
and ring size 1. `--mp-geometry diag` is rejected for this mode. Existing v1/GRASS
choices retain their contracts. Algorithm state identifies `align_new` separately
and refuses a v1 state; this adds no full optimizer/RNG resume support.

Only remove `--dry-run` on the actual allocated GPU with models/data ready and a
new run ID. No training was launched by this port. Full-response head geometry
adds full-vocabulary work; B200/FSDP integration, long-response memory/throughput
and scientific efficacy remain unqualified.

## Verification contract

The oracle executes the donor `_align_loss` and `_grass_chunk_assignment` method
bodies, plus separately loaded original copies of `_mp_opd_align_new.py` and
`_mp_opd_head_pair.py`. Existing v1 Gram, credit, chunk and atom helpers are frozen
shared dependencies, verified by source hashes; the oracle never calls the new
adapter. Source-method fixtures were checked against both the local donor and
the fetched GitHub revision. Kernel changes are LF normalization only.

CPU tests compare complete adapter loss and every student gradient for both chunk
sources, final softcap and rho endpoints. A constructed cross-chunk counterexample
activates the guard and checks the resulting loss/head gradient against the donor.
An independent autograd calculation checks full-response head norms/dot including
bias, softcap and different vocabulary tile widths. Other tests cover teacher
stop-gradient, microbatch normalization, degenerate/invalid geometry, argument
guards and v1 state rejection. Existing Strict/SimCT/MP regressions run alongside.

Use an existing CPU Torch environment, with no package installation:

```bash
TORCH_COMPILE_DISABLE=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
```

The [CPU receipt](ALIGN_NEW_VALIDATION.json) records 65 passing tests. Across
seven FP32 donor comparisons, maximum loss error is `0.0` and maximum gradient
error is `5.960464477539063e-08`. A separate CPU BF16 autocast test checks that
geometry reconstructs the same logits used for credits; it is not a B200 test.
