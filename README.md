# ctopd

Development repository for `sontungkieu`, imported from the authors' [Cross-Tokenizer-OPD code](https://anonymous.4open.science/r/Cross-Tokenizer-OPD/) for [arXiv:2610.08448v1](https://arxiv.org/abs/2610.08448v1). This is an independent working copy; the original authors retain authorship of the imported implementation.

The initial import preserves 84 source/configuration files. See [upstream provenance](docs/UPSTREAM.md) and the exact [file hashes](docs/upstream-source-manifest.json). Dataset and figure directories were excluded from this source snapshot. Model weights, datasets, runtime archives and experiment outputs belong outside Git.

Start editing directly in `kdflow/algorithms/`, `kdflow/token_alignment.py`, `kdflow/trainer/`, and `run/`. The two example launchers use **Qwen2.5-7B-Instruct -> Llama-3.2-3B-Instruct on 8 GPUs**; update paths and qualify the requested topology before training.

Use the existing [qualified B200 runtime on Hugging Face](https://huggingface.co/datasets/codemaivanngu/cross-tokenizer-opd-b200-runtime) with a source overlay rather than reinstalling dependencies after each code edit. See [B200 development instructions](docs/RUNTIME_B200.md).

On a new company node, run the standalone `ensure_runtime.sh`: it restores the pinned shared archive into node-local `/opt/ctopd-b200` if absent, otherwise checks and reuses the existing installation. Ubuntu 22.04/glibc 2.35 automatically uses a verified 30.6 MB compatibility overlay; Ubuntu 24.04/glibc 2.39 retains the original runtime. It preserves unknown/broken installations, uses no `pip`, and starts no training. See [compatibility requirements and evidence](docs/COMPATIBILITY.md) and the runtime instructions for shared-path placement, GPU visibility, and `--install-only` before allocation.

For a full-workload, one-run-per-B200 adaptation, use [the single-B200 launcher](docs/SINGLE_B200.md). It keeps 512-response rollout/optimizer batches, 8,192 generated tokens, 100 training iterations and 32-sample MATH500 evaluation. There is no reduced-workload mode. Single-B200 full training and paper metric agreement remain unvalidated.

An opt-in [MP-OPD port](docs/MP_OPD.md) adds Atomic, Fixed1–5, Random1–2/2–5,
GRASS-Chunk and ALIGN as separate modes. The existing Strict algorithm sources
remain unchanged. The new modes use sampled atom credits rather than Strict KL;
they have CPU verification, with B200/full training qualification still pending.

A separate [author-code SimCT baseline](docs/SIMCT.md) uses `--algorithm simct`
in the same launcher. It preserves the pinned released implementation, with
original-source loss/gradient parity tests and an explicit paper/code distinction.

## Original project documentation

This repository provides the implementation and example training scripts for `byte_align_ctkd` and `topk_byte_align_ctkd`.

The overview figure is available in the [original repository](https://anonymous.4open.science/r/Cross-Tokenizer-OPD/README.md); assets are not included in this source import.

## Environment

Our experimental environment was created on top of the [KDFlow Docker image](https://hub.docker.com/r/songmzhang/kdflow?tag=sgl0.5.12_torch2.11_cu129), using the following tag:

```bash
docker pull songmzhang/kdflow:sgl0.5.12_torch2.11_cu129
```

[`requirements.txt`](requirements.txt) records the package versions used in our experiments. Use the Docker image as the starting point and match any additional dependencies to this file. It includes CUDA-specific builds and system-provided packages, so it is an environment snapshot rather than a standalone installation specification for arbitrary hosts.

Inside the prepared container, run from the repository root:

```bash
python -m pip install -e . --no-deps
```

## Example training

Both scripts perform on-policy distillation from **Qwen2.5-7B-Instruct** to **Llama-3.2-3B-Instruct** and are configured for **one node with 8 GPUs**.

Before running, update the paths in each script or mount the files at the expected locations:

- Repository: `/workspace/Cross-Tokenizer-OPD`
- Student checkpoint: `/workspace/models/Llama-3.2-3B-Instruct`
- Teacher checkpoint: `/workspace/models/Qwen2.5-7B-Instruct`
- Training data: `/workspace/data/Train/math_code_20k/train.parquet`
- Evaluation data: `/workspace/data/math500/test.jsonl`

The original repository provides data under `data/`; this source-only import excludes that directory. Prepare data and model checkpoints separately at the runtime host, then update the paths above. WandB logging is disabled by default (`--use_wandb False`). To enable it, set `--use_wandb True`, add `--wandb_project <project_name>`, and configure your WandB account.

Run the byte-alignment example from the repository root:

```bash
bash run/byte_align_ctkd/qwen25_7b_to_llama32_3b_byte_align_ctkd.sh
```

Run the top-k example (`--ctkd_topk 16`):

```bash
bash run/topk_byte_align_ctkd/qwen25_7b_to_llama32_3b_topk_byte_align_ctkd.sh
```

Training settings can be adjusted directly in the scripts. Checkpoints, evaluation results, and training logs are written under `outputs/<TRAIN_NAME>/`.
