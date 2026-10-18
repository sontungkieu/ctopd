# Cross-Tokenizer-OPD

This repository provides the implementation and example training scripts for `byte_align_ctkd` and `topk_byte_align_ctkd`.

![Overview of cross-tokenizer alignment during training, the effect of mismatch supervision, and shared-vocabulary probability mass](assets/overview.png)

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

The data files are included under `data/`; model checkpoints must be provided separately. WandB logging is disabled by default (`--use_wandb False`). To enable it, set `--use_wandb True`, add `--wandb_project <project_name>`, and configure your WandB account.

Run the byte-alignment example from the repository root:

```bash
bash run/byte_align_ctkd/qwen25_7b_to_llama32_3b_byte_align_ctkd.sh
```

Run the top-k example (`--ctkd_topk 16`):

```bash
bash run/topk_byte_align_ctkd/qwen25_7b_to_llama32_3b_topk_byte_align_ctkd.sh
```

Training settings can be adjusted directly in the scripts. Checkpoints, evaluation results, and training logs are written under `outputs/<TRAIN_NAME>/`.
