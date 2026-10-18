# Upstream provenance

- Paper: https://arxiv.org/abs/2610.08448v1
- Authors' source: https://anonymous.4open.science/r/Cross-Tokenizer-OPD/
- Retrieval timestamp: 2026-10-17T19:28:35.738288+00:00
- Imported files: 84 (78 Python files), checked against SHA-256 before import.
- Snapshot manifest hash: `93cef1dfa18bcd5fc66cf80347557382efcd7e8955c3f6b464435711e82bf49b`.
- The source service did not expose an upstream Git commit. The local import commit starts a new history and does not claim to preserve upstream Git history.
- `docs/upstream-source-manifest.json` records original byte sizes and hashes. It describes the initial import; later development commits may intentionally change those files.

The initial commit preserves the imported implementation. Local additions provide documentation, line-ending attributes, a single-B200 launcher, a standalone node-local runtime setup helper and offline tests. Imported `kdflow/` code and the two original training scripts remain unchanged. Datasets and figures were excluded from the source snapshot, and no runtime archives, model weights, credentials, account metadata or experiment outputs are included.

## License metadata

The original `pyproject.toml` declares MIT and refers to `LICENSE`, `THIRD_PARTY_NOTICES.md`, and `LICENSES/*.txt`. Those files were absent from the retrieved source listing. This working copy preserves that metadata without inventing missing license text or changing third-party ownership. Consult the original authors for the missing license/notice files before making independent licensing claims.

## Validation boundary

Initial import verification checks source hashes, Python syntax and Bash syntax only. This is not an end-to-end training run. The separate runtime qualification covers six synthetic probes on one Modal B200; it does not validate production training, company-node compatibility or multi-GPU behavior.
