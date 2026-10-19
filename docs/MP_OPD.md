# Isolated MP-OPD port: Atomic, Fixed, Random, GRASS-Chunk and ALIGN

These are experimental on-policy modes, ported from Tung's SimCT/MP-OPD checkout.
They are not aliases for Strict full/top-16, and their results must not be reported
as reproduction of the CTOPD paper's Strict objective or the full SimCT recipe.
Models, data, teacher warm start, optimizer, rollout and evaluation still come from
the CTOPD launcher. The initial adapter requires one student rank, ring size 1,
`kd_ratio=1`, `kd_temperature=1`, and `kd_loss_fn=rkl`.

## Available presets

| `tools/launch_single_b200.py --algorithm` | Registered algorithm | Intervention |
| --- | --- | --- |
| `mp-atomic` | `mp_opd_atomic` | Original atom credits |
| `mp-fixed1` through `mp-fixed5` | `mp_opd_fixed` | Hard pooling over 1 through 5 atoms |
| `mp-random1-2` | `mp_opd_random` | Hard pooling, group length sampled uniformly from [1,2] |
| `mp-random2-5` | `mp_opd_random` | Hard pooling, group length sampled uniformly from [2,5] |
| `mp-grass-chunk` | `mp_opd_grass_chunk` | Risk-adaptive shrinkage within supplied chunks; no DP |
| `mp-align` | `mp_opd_align` | Donor ALIGN v1 cyclic coefficient-space PCGrad within the same chunks |
| `mp-align-new` | `mp_opd_align_new` | Local ALIGN plus full-response head-descent guard; see [ALIGN_NEW.md](ALIGN_NEW.md) |
| `mp-trust-r` | `mp_opd_trust_r` | Strict plus mismatch scaled once per response |
| `mp-trust-b` | `mp_opd_trust_b` | Strict plus mismatch scaled once per complete optimizer batch |
| `mp-trust-p` | `mp_opd_trust_p` | Project opposing mismatch component through a complete-batch strict scale |

See [TRUST_PAPER.md](TRUST_PAPER.md) for the two-pass B/P actor contract and the
separate `simct_paper` score reconstruction. TRUST uses strict/mismatch atom types,
not the `xtoken` chunk partition used by ALIGN/GRASS.

Fixed/random groups tile each response independently. The final tail may be
shorter than the requested length/minimum. Randomness uses the configured seed
and response token IDs, independent of microbatch row/rank and global Python RNG.
This intentionally improves grouping stability over the donor's row-index seed;
it is a documented integration difference, not identical donor RNG replay.

## Run and dry-run

From this checkout, print the full workload without importing ML libraries,
creating a run, or using a GPU:

```bash
python3 tools/launch_single_b200.py --algorithm mp-align \
  --runtime /opt/ctopd-b200-u22 --dry-run
python3 tools/launch_single_b200.py --algorithm mp-grass-chunk \
  --runtime /opt/ctopd-b200-u22 --dry-run
python3 tools/launch_single_b200.py --algorithm mp-fixed5 --dry-run
python3 tools/launch_single_b200.py --algorithm mp-random2-5 --dry-run
```

Remove `--dry-run` only in the actual allocated B200 environment, exposing the one
GPU assigned to the run through `CUDA_VISIBLE_DEVICES`. Models/data must already
exist at the launcher's paths. Each run needs a fresh run ID. These presets retain
512 rollout/optimizer responses, max generation 8192, 100 iterations, and step-0
plus every-10-step MATH500 evaluation with 32 responses/problem. Defaults for the
two Strict presets remain unchanged. Evaluation is still MATH500, not the full
paper benchmark suite.

GRASS-Chunk and ALIGN default to `--mp-chunk-source xtoken`, using the same
vendored tokenizer/string aligner as the donor. This use does not need a learned
vocabulary projection. `--mp-chunk-source run --mp-chunk-run-length 2` explicitly
chooses the fixed-run ablation instead. `--mp-geometry diag` is a separate diagonal
geometry ablation; the default is `exact_head`. ALIGN here is v1 `align`, not
`align_new` or a full-Transformer/global-descent guard. The separately ported
`mp-align-new` has a full-response **head** guard; it still does not measure the
full Transformer gradient. The donor does not name these modes v2/v3.

## Scientific and numerical contract

Each response is byte-checked and partitioned into the donor's minimal synchronized
atoms. Terminal EOS is excluded from this mode family's credits (Strict's EOS
handling is unchanged). Atom credit is the detached teacher-minus-current-student
log-score, divided by student token count. The student loss is detached effective
credit times differentiable atom NLL, divided by the existing shared optimizer-window
token normalizer. It is a sampled path-credit surrogate and may be negative; it is
not the directly enumerated shared-vocabulary KL used by Strict.

GRASS reuses the corrected donor span-wide distortion, noise estimator and exact
LM-head Gram. The noise estimator updates per response and excludes adjacency
across chunk boundaries. Weighted signed credit is checked after dtype conversion;
evaluation uses the stored noise estimate without updating it. An invalid result
fails rather than silently switching to Atomic. ALIGN does not
claim credit conservation: its intervention is deterministic direction-conflict
projection. All geometry is detached. Geometry flags restore process-global FP32
settings; disabling `MP_OPD_GRASS_EXACT_FP32` is rejected for these two modes.

The adapter applies the configured student/teacher final-logit softcap to raw
linear heads. Token tiles use checkpointed log-prob computation; changing
`chunked_loss_size` does not set research chunk boundaries. Exact-head geometry
still materializes per-response logits and an atom Gram, so long-response GPU
memory/throughput remain unqualified. No new wheels or native extension are added.

Invalid byte alignment fails with the failure reason, with no automatic fallback.
Empty content responses produce zero contribution. Singleton alignment chunks
leave GRASS/ALIGN unchanged; inspect singleton fractions and coefficient-change
metrics before claiming an intervention occurred.

`state_dict`/`load_state_dict` expose algorithm state and reject a changed mode or
MP-OPD contract. They do not add full training resume: existing HF model exports do
not save optimizer/RNG/GRASS noise state. The launcher continues to refuse existing
run directories. Fresh-run comparisons only are supported by this integration.

## Isolation, provenance and verification

`kdflow/algorithms/mp_opd.py` is the new adapter. Donor kernels live only in
`kdflow/algorithms/_mp_opd_port/` and are imported lazily after selecting an MP mode.
The three existing algorithm files, byte alignment, RKL and tiled loss are
byte-identical to the pre-port checkpoint. Shared edits only register names,
validate the new flags, allow them through the on-policy CLI, attach teacher
softcap metadata for MP modes, and provide launcher presets.

`MP_OPD_SOURCE.json` records the donor commit, every source/ported file hash,
line-ending/import changes, and frozen Strict source hashes. The donor repository's MIT
notice and the NVIDIA aligner's Apache-2.0 notice/license are retained alongside
the vendored files. The donor checkout's unrelated staged diagnostic files were
not modified or committed here.

Run CPU verification using an existing compatible PyTorch environment; do not
install dependencies merely to run this command:

```bash
TORCH_COMPILE_DISABLE=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
```

The tests cover donor Atomic loss/gradient parity, Fixed1 identity, fixed/random
coverage, RNG independence, per-response/microbatch normalization, numerical tile
invariance, GRASS identities/conservation, explicit-autograd head-Gram parity with
softcapping, nontrivial ALIGN projection, native/run chunk backward, source hashes,
and existing Strict loss/gradient regression. A Python environment without torch
skips tensor tests explicitly; it is not a full verification pass.

The checked CPU environment and commands are recorded in
[MP_OPD_VALIDATION.json](MP_OPD_VALIDATION.json).
CPU tests establish implementation checks only. B200 execution, FSDP runtime
integration, full-size memory/throughput, training resume and scientific efficacy
have not been qualified for the new modes. No training or GPU job is launched by
the port or its tests.
