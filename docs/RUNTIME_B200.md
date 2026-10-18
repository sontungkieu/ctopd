# Develop against the existing B200 runtime

Runtime mirror: https://huggingface.co/datasets/codemaivanngu/cross-tokenizer-opd-b200-runtime

- Artifact revision: `6058422a8319cbe03958bf5d52750293e4e0da65`.
- Downloader revision: `93a118a96df88fe7dd0852b50d5b62616b701aa8`.
- Archive: `ctopd-b200-cu129-v4/runtime.tar.gz`, 16,515,065,867 bytes.
- SHA-256: `084c321c0b7af1cd3a36d12a595c623567d4a37525aa0eecbc3933aa7f9b15c3`.

Download/install directly on the target host using the pinned HF instructions. The archive requires Linux x86_64, glibc >= 2.39, g++ for JIT, and a B200 driver compatible with CUDA 12.9. Allow about 35 GiB free for parallel download/assembly or 60 GiB including installation. Model weights and datasets are separate.

Once installed, point `RUNTIME` at the actual extracted prefix and run source from this checkout:

```bash
cd /absolute/path/to/ctopd
RUNTIME=/absolute/path/to/installed/ctopd-b200
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
bash "$RUNTIME/python-b200.sh" -m kdflow.cli.train_kd_on_policy --help
```

The CLI imports CUDA dependencies and should be checked inside the allocated GPU environment. It is not a CPU-only test. Code/configuration edits do not require rebuilding dependencies; requalify runtime changes that affect CUDA, Torch, native extensions or ABI compatibility.

Before training, update the example scripts' source/model/data/output paths and actual GPU allocation. Their current hard-coded source prefix is `/workspace/Cross-Tokenizer-OPD`; change it to this checkout so they load your edits. The original scripts assume one node with eight GPUs. No training job is launched by these instructions.
