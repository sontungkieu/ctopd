# One full Qwen -> Llama run per B200

`tools/launch_single_b200.py` prepares the original workload for one exclusively allocated B200. It has no toy/canary workload and does not install packages or download models/data. `--dry-run` prints configuration only; a normal invocation starts the real 100-iteration run, including full MATH500 evaluation at step 0 and every 10 iterations.

Separate experimental MP-OPD presets (Atomic, Fixed1-5, Random1-2/2-5, GRASS-Chunk and ALIGN) are documented in [MP_OPD.md](MP_OPD.md). They retain the workload below but change the objective; they are not Strict reproduction modes. Their CPU checks do not qualify B200/FSDP execution or full-training memory requirements.

The separate [SimCT author-code preset](SIMCT.md), `--algorithm simct`, uses the same common workload. It adds no SFT warmup; it is not the original SimCT end-to-end recipe.

## Preserved training and evaluation settings

| Setting | Value |
| --- | --- |
| Teacher | Qwen2.5-7B-Instruct, full checkpoint, no quantization |
| Student | Llama-3.2-3B-Instruct, full parameter training, no extra SFT |
| Algorithm | Strict top-16 by default; `--algorithm strict-full` selects the full shared-vocabulary baseline |
| KD | Reverse KL, weight 1, temperature 1 |
| Global rollout / optimizer batch | 512 / 512 responses |
| Responses per prompt | 1 |
| Training iterations | 100 |
| Prompt / generated / combined maximum tokens | 2,048 / 8,192 / 10,240 |
| Sampling | Temperature 1, top-p 0.95 |
| Optimizer | AdamW, LR 1e-6 constant, betas (0.9, 0.98), weight decay 0, grad clip 1 |
| Computation | BF16, gradient checkpointing, packed samples, loss chunk 2,048 |
| Checkpoint / MATH500 evaluation interval | 10 / 10 iterations, plus initial evaluation |
| MATH500 sampling | 32 responses per problem, 8,192 generated tokens, same temperature/top-p |

The launcher explicitly uses seed 42, matching the original launcher's code default. This does not establish the seed used for the paper's reported tables. Model commits and dataset hashes used by the original paper were not supplied in the imported launchers; pin and record the revisions actually downloaded for this reproduction.

## Changed execution topology and limits

- One node / one B200; one teacher engine with TP=DP=PP=1; one student rollout engine with TP=1; student FSDP2 world size 1.
- The original launcher uses 8 GPUs, teacher DP=4 and TP=2, and 8 rollout engines. Those GPU counts are not portable to one B200 by changing only `num_gpus_per_node`.
- Teacher, rollout and student share the GPU in successive phases using the existing sleep/offload implementation. This is full-weight training, not LoRA or a smaller model.
- Micro-batch remains 4 by default, giving 128 gradient-accumulation microsteps per update. `--micro-batch 1` or `2` changes execution granularity while retaining global batch 512. Global batch is never reduced.
- Generation/evaluation request concurrency defaults to 8 instead of the source defaults of 512/256. This caps simultaneous requests, not the number of samples. Preprocessing uses 8 CPU workers instead of 32. The evaluation request timeout is disabled by default (`--eval-timeout 0`) to avoid imposing the original 2,400-second request limit on slower single-GPU execution.
- The launcher starts a fresh local Ray instance, keeps output directories separate and never attaches to an unrelated Ray cluster. Local hostname/IP and loopback connections are exempted from the inherited HTTP proxy for Ray/SGLang; external network proxy settings are retained.

The inspected code divides each micro-batch's summed loss by average response-token count, divides backward loss by gradient accumulation, and averages distributed gradients. On fixed responses, this preserves the batch token-weighted objective across these world/micro-batch sizes in exact arithmetic. A standard-library scalar regression exercises the actual preparation/backward methods with uneven response lengths for (world, micro) = (8,4), (1,4), (1,1). This checks normalization algebra; it does not test CUDA tensors, FSDP gradient parity or training trajectories.

One B200 therefore defines an **unvalidated topology adaptation**, not a bitwise reproduction of eight H20 GPUs. Different random-number consumption, request scheduling, tensor-parallel arithmetic and BF16 reduction order can change sampled responses and final scores. The full workload must still be run and compared with the paper before claiming agreement. No runtime or throughput measurement of this full configuration is available yet.

CPU RAM and `/dev/shm` also matter: the teacher forwards a whole rollout batch and retains response hidden states in CPU/Ray memory. One teacher DP engine concentrates work previously split over four. Existing CPU weight/optimizer offload adds host-memory pressure. Lowering request concurrency alone does not bound the retained 512-response hidden-state memory. Do not infer host-memory sufficiency from B200 VRAM alone.

## Required files

With `ROOT=/workspace/storage-shared/nlp/tungks/ctopd_all`, prepare:

```text
/opt/ctopd-b200/python-b200.sh          # node-local; restore using ensure_runtime.sh
$ROOT/models/Qwen2.5-7B-Instruct/       # complete official HF snapshot + tokenizer
$ROOT/models/Llama-3.2-3B-Instruct/     # complete official HF snapshot + tokenizer
$ROOT/data/Train/math_code_20k/train.parquet
$ROOT/data/math500/test.jsonl
```

The launcher defaults to `CTOPD_RUNTIME_PREFIX` when set, otherwise `/opt/ctopd-b200`. Use `--runtime /actual/installed/prefix` for an existing alternative installation, including the earlier `$ROOT/runtime` layout. Run the [runtime restore/check helper](RUNTIME_B200.md) on every new node. The runtime tar.gz alone is not an installed prefix. Models/data must be downloaded directly on the runtime host; the Llama HF token must belong to an account with approved model access.

## Print the complete plan, then run it unchanged

From this checkout on the company host:

```bash
python3 tools/launch_single_b200.py --run-id qwen-llama-top16-s42 --dry-run
```

Inside a pod exposing exactly the single B200 allocated to this run:

```bash
set -euo pipefail
python3 tools/launch_single_b200.py --run-id qwen-llama-top16-s42
```

On a shared multi-GPU host, set `CUDA_VISIBLE_DEVICES` to the **allocated GPU UUID** for the command; do not select an arbitrary idle device. The launcher checks that the runtime sees exactly one B200 before starting training. Visibility is not proof of allocation: obtain that allocation first. Separate pods are preferable for independent runs.

The launcher's preflight checks model/tokenizer files, all referenced safetensors shards, GPU visibility, the 20k-row source training pool and all 500 MATH500 problems. It then records source commit, effective arguments, data SHA-256 and preflight observations in `$ROOT/runs/<run-id>/launch.json`. It refuses a dirty source checkout, an existing output directory, or implicit resume from an unrelated checkpoint. Checkpoints/results go under that run's `checkpoints/`. Secrets and the full process environment are not recorded. It does not verify all model tensor hashes or independently requalify the installed runtime; the archive hash in the plan is the expected identity.

To collect the full shared-vocabulary comparison, allocate a different GPU/run and use a distinct run-id:

```bash
python3 tools/launch_single_b200.py --algorithm strict-full --run-id qwen-llama-full-s42
```

These are two independent complete runs; never run both simultaneously on the same allocated GPU. Each invocation also performs the initial full MATH500 evaluation before the first train iteration; lack of optimizer logs during that evaluation does not mean training is stalled.

## Compare the right quantities

For MATH500, compare `eval/accuracy * 100` with the paper's accuracy over 32 samples, **not** `eval/pass@32`. Inspect the saved generation results for missing/failed requests before accepting a score. The current launcher's evaluation covers MATH500 only. Reproducing the paper's Math, Code and Full aggregate scores also needs the other six math benchmarks and all three code benchmarks, the same step-100 checkpoint, 32/8 samples respectively, and matching scoring scripts. This launcher alone does not implement that complete benchmark matrix.

## Validation performed locally

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p test_single_b200.py -v
```

Six lightweight tests cover the full-workload contract, one-GPU topology, flag compatibility with the actual argument dataclasses, safe dry-run, preservation of existing runs, and token-normalization algebra. No GPU job, model download, company installation or full training was executed while preparing this launcher.
