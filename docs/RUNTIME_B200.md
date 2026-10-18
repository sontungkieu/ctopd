# Develop against the existing B200 runtime

Runtime mirror: https://huggingface.co/datasets/codemaivanngu/cross-tokenizer-opd-b200-runtime

- Artifact revision: `6058422a8319cbe03958bf5d52750293e4e0da65`.
- Downloader revision: `93a118a96df88fe7dd0852b50d5b62616b701aa8`.
- Archive: `ctopd-b200-cu129-v4/runtime.tar.gz`, 16,515,065,867 bytes.
- SHA-256: `084c321c0b7af1cd3a36d12a595c623567d4a37525aa0eecbc3933aa7f9b15c3`.

Download/install directly on the target host using the pinned HF instructions. The archive requires Linux x86_64, glibc >= 2.39, g++ for JIT, and a B200 driver compatible with CUDA 12.9. Allow about 35 GiB free for parallel download/assembly or 60 GiB including installation. Model weights and datasets are separate.

## Restore/check on every new node

`ensure_runtime.sh` is self-contained and can be copied from this checkout into `/workspace/storage-shared/nlp/tungks/ctopd_all/ensure_runtime.sh`. The archive stays on shared storage; the default installed prefix `/opt/ctopd-b200` belongs to the current node/pod. Shared files do not imply that another node has the runtime installed.

```bash
ROOT=/workspace/storage-shared/nlp/tungks/ctopd_all
cp ensure_runtime.sh "$ROOT/ensure_runtime.sh"  # run once from this checkout
bash "$ROOT/ensure_runtime.sh"
```

Run the last line on each new node, inside a pod exposing the one B200 allocated for this check. With no GPU allocated yet, use `--install-only`; it reports `RUNTIME_INSTALLED_UNCHECKED`, not GPU readiness. With an existing runtime, `--check-only` checks without extracting. `--help` lists options.

The helper automatically finds either `$ROOT/ctopd-b200-cu129-v4/runtime.tar.gz` (the manually downloaded shard layout) or `$ROOT/runtime.tar.gz`. It verifies the pinned archive size/SHA-256 before extraction, checks Linux/glibc/g++, extracts safely to a new staging directory under the target filesystem, then renames to the final prefix. Allow at least 45 GiB free on that node-local filesystem. It never downloads dependencies, builds a runtime, launches training, or replaces an existing runtime that it cannot identify. Failed extraction keeps staging evidence. If `/opt` is not writable, choose an explicit node-local location:

```bash
export CTOPD_RUNTIME_PREFIX="/var/tmp/ctopd-b200-$UID"
bash "$ROOT/ensure_runtime.sh"
```

Concurrent setup calls use a lock for the same prefix. Reuse checks its archive-identity receipt and hashes of the wrapper, portable Python and nvcc; it does not rehash every installed dependency or reread the 16.5 GB archive. Default readiness additionally verifies the pinned Python/Torch/Transformers/SGLang/FlashAttention/FlashInfer versions, NVRTC headers, imports and a one-element CUDA memory roundtrip on the allocated B200. This is a basic runtime check, not full kernel qualification or a training result. Existing incompatible/unmanaged prefixes are preserved for manual inspection; there is no recursive-delete fallback.

The script can also execute a Python command after readiness:

```bash
bash "$ROOT/ensure_runtime.sh" -- -c 'import torch; print(torch.__version__)'
```

This is a fresh implementation for CTOPD's CUDA 12.9 bundle. The historical GOLD/SimCT helper used different CUDA 13 prefixes and a destructive repair fallback; those paths and that fallback are not copied into CTOPD. Live installation on the company node has not been executed from this chat. Offline archive-fixture tests and Bash syntax checks are separate from the real node readiness result.

## Use source without rebuilding dependencies

Once installed, point `RUNTIME` at the actual extracted prefix and run source from this checkout:

```bash
cd /absolute/path/to/ctopd
RUNTIME="${CTOPD_RUNTIME_PREFIX:-/opt/ctopd-b200}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
bash "$RUNTIME/python-b200.sh" -m kdflow.cli.train_kd_on_policy --help
```

The CLI imports CUDA dependencies and should be checked inside the allocated GPU environment. It is not a CPU-only test. Code/configuration edits do not require rebuilding dependencies; requalify runtime changes that affect CUDA, Torch, native extensions or ABI compatibility.

Before training, update the example scripts' source/model/data/output paths and actual GPU allocation. Their current hard-coded source prefix is `/workspace/Cross-Tokenizer-OPD`; change it to this checkout so they load your edits. The original scripts assume one node with eight GPUs. No training job is launched by these instructions.
