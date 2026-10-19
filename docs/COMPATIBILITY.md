# Required node compatibility targets

Both baselines are requirements for the CTOPD runtime, recorded at the user's request on 2026-10-19:

| Required target | Evidence and current boundary |
| --- | --- |
| Ubuntu 22.04.5 LTS, glibc 2.35, Linux x86_64, B200 | User supplied OS/libc evidence from `anhhbt-quant-0-0`. The fixed runtime passed all six infrastructure probes on a real Modal B200 with Ubuntu 22.04/glibc 2.35, plus a CPU router protocol check. The actual company node still needs the delivered helper check. |
| Ubuntu 24.04, glibc 2.39, Linux x86_64, B200 | Original v4 passed six B200 infrastructure probes. The original artifact/path is retained and passed a fresh CPU Torch/Transformers/router-import regression on glibc 2.39. No new full-training result is implied. |

## Root cause of the Ubuntu 22.04 failure

The v4 package copied host libraries from the Ubuntu 24 author image. A read-only ELF audit found 99 bundled host libraries with GLIBC requirements above 2.35, plus `sglang_router/sglang_router_rs.abi3.so` requiring GLIBC 2.39. All other audited runtime ELF objects had requirements no higher than 2.35. Changing a shell guard cannot change these ELF requirements.

The overlay retains Python/Torch/CUDA versions and the research implementation. It substitutes a glibc-2.35 C++/gcc/zlib set and the Python/native router package from one pinned `sglang-router==0.3.2` manylinux wheel. The original image's Python router API differs despite the same version label: replacing only the binary fails on the unsupported `json_log` constructor keyword. The qualified overlay includes the matched Python files. CTOPD's actual round-robin startup, worker registration and `/generate` forwarding protocol passed; the research code is unchanged. Host libc is never bundled or replaced.

## Immutable overlay and evidence

- HF revision: `3b1d90bc044874dc20c65308ec72841fc3db22e9`.
- File: `compat/glibc235-v2/overlay.tar.gz`, 30,628,766 bytes.
- SHA-256: `21cc38ecdfc2f61714922127b67f3a0be89bf70bb2178b8fda8d3659e14f76c7`.
- [Full qualification receipt](https://huggingface.co/datasets/codemaivanngu/cross-tokenizer-opd-b200-runtime/blob/3b1d90bc044874dc20c65308ec72841fc3db22e9/compat/glibc235-v2/qualification.json) and adjacent overlay manifest contain image IDs, dependency hashes and logs.
- The Ubuntu 22 GPU suite passed in 80.143386344 seconds: BF16, imports, packed FlashAttention forward/backward, FlashInfer, FSDP2/top-16 reverse-KL, patched SGLang hidden states/logprobs/sleep-wake. It used synthetic fixtures, not paper training. After the pass markers, Python emitted a multiprocessing resource-tracker cleanup `ChildProcessError`; the process still exited zero. Full company training remains unverified.
- The helper restored the complete original archive plus overlay on clean Ubuntu 22 in 399.609342267 seconds, reused that install in 0.244060964 seconds, and passed actual Torch CPU and router protocol checks. The complete CPU path took 412.366908248 seconds. [Restore receipt](https://huggingface.co/datasets/codemaivanngu/cross-tokenizer-opd-b200-runtime/blob/3f6ca2db515415208742f0a1846a761ab361728d/compat/glibc235-v2/restore-qualification.json) tests installation independently of the separately qualified B200 image.

The helper selects the overlay for glibc 2.35 through 2.38 and keeps the original v4 path for glibc >= 2.39. Only 2.35 and 2.39 have live baseline evidence. It caches the overlay once on shared storage, checks its pinned bytes/hash, quarantines original host libraries inside the new installation, and records all substituted file hashes. The base archive is unchanged and does not need another download.

Preserve the original Ubuntu 24 path. The helper must require the verified overlay on older supported glibc hosts, rather than treating the unpatched v4 package as compatible after lowering a version check. Existing incompatible installs are preserved. The current helper automatically chooses `/opt/ctopd-b200-u22` when a known original v4 occupies `/opt/ctopd-b200` on an older host; explicit prefix overrides still fail if incompatible. No active runtime is patched in place.

The exact company node is a reported reproduction target, not a lease or proof of available GPU capacity. Live free-GPU claims from earlier transcripts are not allocation evidence.
