# Invalid decoded responses in MP-OPD

The 2026-10-19 company logs showed `replacement_character` in Fixed1 and
`normalization_mismatch` in GRASS-Chunk and ALIGN. Their root exceptions occurred
in the shared MP adapter; the displayed SIGTERM messages accompanied Ray cleanup.
Those logs do not contain the decoded strings needed to identify the underlying
generation/retokenization discrepancy. This patch mitigates the crash by excluding
responses that cannot be scored safely on synchronized byte spans.

## Policy

All MP modes, including ALIGN-NEW and TRUST-R/B/P, exclude the complete response
for `replacement_character`, `normalization_mismatch` or `empty_response`.
Student and teacher hidden-state offsets still advance by their full response
lengths. No text is rewritten and no alternative Atomic loss is substituted.
Other failures, including unsupported added tokens, remain fatal. The vendored
atomizer and all geometry/loss kernels are unchanged.

The existing shared optimizer-window denominator is preserved, including excluded
tokens. This avoids changing the weighting of valid responses according to how
they happen to be grouped into microbatches. Exclusion changes the effective
sample distribution and reduces gradient magnitude relative to processing every
token; it is not mathematical equivalence to training on the original batch.
An entirely excluded microbatch returns connected zero loss. That alone does not
guarantee unchanged optimizer state or weights across an entire optimizer update.

Metrics use the prefix `distill/mp_opd_`:

- `attempted_responses`, `valid_responses`, `skipped_responses`;
- `skipped_replacement_responses`, `skipped_normalization_responses`,
  `skipped_empty_responses`;
- `skipped_replacement_fraction`, `skipped_normalization_fraction`;
- `skipped_student_loss_tokens`, `skipped_teacher_loss_tokens`,
  `skipped_student_token_fraction`.

These are microbatch counts/fractions. Existing training logging aggregates
microbatches; a displayed count may therefore be an average rather than a run
total. Record exclusions when interpreting comparisons, especially if their
frequency differs by method. The policy does not affect `simct`, `simct_paper`,
or Strict.

## Verification and provenance

`tests/test_invalid_response.py` checks whole-response exclusion, hidden-state
offsets, the original denominator, valid-response loss/all student gradients,
all-excluded backward, transient UTF-8 prefix recovery, failure guards, and
TRUST-B/P calibration/replay. Historical source manifests are retained;
`MP_OPD_INVALID_RESPONSES.json` explicitly records the patched adapter hash.
CPU checks do not qualify B200/FSDP execution or full training. The company logs
were inputs to diagnosis, not successful execution evidence for this patch.

Only source changes are needed; the runtime archive does not need rebuilding.
Use a fresh source checkout and a new run ID for failed runs. Existing Ray actors
carry a packaged source snapshot, so updating files cannot repair an already
running actor. Preserve active jobs' checkouts and prior run outputs.
