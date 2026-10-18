import os
import math
from dataclasses import dataclass, field
from typing import Optional


def resolve_max_steps(
    configured_max_steps: int,
    num_epochs: int,
    steps_per_epoch: int,
) -> int:
    """Resolve the explicit step limit or the epoch-derived training length."""
    if steps_per_epoch <= 0:
        raise ValueError(
            "Training requires at least one full global step per epoch. "
            "Reduce the global batch size or provide more training samples."
        )
    if configured_max_steps > 0:
        return configured_max_steps
    return math.ceil(num_epochs * steps_per_epoch)


@dataclass
class TrainingArguments:
    """ Arguments for training."""
    
    num_nodes: int = field(
        default=1,
        metadata={"help": "The number of nodes for the student/teacher model (only for on-policy KD)."}
    )
    num_gpus_per_node: int = field(
        default=8,
        metadata={"help": "The number of gpus per node for the student/teacher model (only for on-policy KD)."}
    )
    num_epochs: int = field(
        default=1,
        metadata={"help": "Number of training epochs."}
    )
    max_steps: int = field(
        default=-1,
        metadata={
            "help": (
                "Maximum global training steps. A positive value overrides "
                "num_epochs; for on-policy KD, one global step is one rollout "
                "iteration."
            )
        },
    )
    train_batch_size: int = field(
        default=128,
        metadata={"help": "Global training batch size."}
    )
    micro_train_batch_size: int = field(
        default=1,
        metadata={"help": "Micro training batch size per GPU."}
    )
    learning_rate: float = field(
        default=1e-6
    )
    lr_warmup_ratio: float = field(
        default=0.05
    )
    min_lr: float = field(
        default=1e-8
    )
    lr_scheduler: str = field(
        default="cosine_with_min_lr"
    )
    max_norm: float = field(
        default=1.0,
        metadata={"help": "Gradient clipping."}
    )
    weight_decay: float = field(
        default=0.0,
        metadata={"help": "Weight decay."}
    )
    adam_betas: str = field(
        default="(0.9, 0.98)"
    )
    eval_steps: int = field(
        default=-1,
        metadata={"help": "Evaluation every n step."}
    )
    save_steps: int = field(
        default=-1,
        metadata={"help": "Save checkpoints every n step."}
    )
    backend: str = field(
        default="fsdp2",
        metadata={
            "choices": ["fsdp2"],
            "help": "Student training backend."
        }
    )
    gradient_checkpointing: bool = field(
        default=False
    )
    gradient_checkpointing_use_reentrant: bool = field(
        default=False
    )
    train_enable_sleep: bool = field(
        default=False,
        metadata={"help": "[Deprecated] Use --enable_sleep instead."}
    )
    enable_sleep: bool = field(
        default=False,
        metadata={"help": "Enable sleep mode for all components (student, teacher, rollout)."}
    )
    full_determinism: bool = field(
        default=False,
        metadata={"help": "Enable reproducible behavior during distributed training."}
    )
    load_checkpoint: bool = field(
        default=False
    )
    ckpt_path: str = field(
        default="./ckpt/checkpoints_distill"
    )
    save_path: str = field(
        default="./ckpt/"
    )
    seed: int = field(
        default=42,
    )
    use_dynamic_bsz: bool = field(
        default=False,
        metadata={"help": "Enable dynamic batch size based on token count per GPU."}
    )
    max_token_len_per_gpu: int = field(
        default=0,
        metadata={"help": "Maximum total token count per micro-batch when use_dynamic_bsz is True."}
    )
    bf16: bool = field(
        default=False
    )
    local_rank: int = field(
        default=int(os.getenv("LOCAL_RANK", -1)),
        metadata={"help": "Local rank for distributed training. Automatically set."}
    )
    chunked_loss_size: int = field(
        default=None,
        metadata={"help": "Token chunk size for chunked loss. If None, chunked loss is disabled."}
    )
    
    def __post_init__(self):
        # Ray actors run from a temporary working directory. Resolve output paths
        # in the driver before the arguments are serialized and sent to actors so
        # relative paths keep their command-invocation semantics.
        self.save_path = self._resolve_driver_path(self.save_path)
        self.ckpt_path = self._resolve_driver_path(self.ckpt_path)

        backend_choices = self.__class__.__dataclass_fields__["backend"].metadata["choices"]
        if self.backend not in backend_choices:
            raise ValueError(
                f"Unsupported training backend '{self.backend}'"
                f"Supported choices: {backend_choices}"
            )

        if isinstance(self.adam_betas, str):
            try:
                cleaned_str = self.adam_betas.strip().strip('()')
                parts = [p.strip() for p in cleaned_str.split(',')]
                if len(parts) != 2:
                    raise ValueError("adam_betas must contain two values.")
                
                self.adam_betas = (float(parts[0]), float(parts[1]))
            except Exception as e:
                raise ValueError(
                    f"Cannot parse '{self.adam_betas}' to valid adam_betas. Make sure the format to be '(beta1, beta2)'。"
                ) from e
        else:
             raise TypeError(f"Expected str for adam_betas, but get {type(self.adam_betas)}")
         
        if self.save_steps <= 0:
            self.save_steps = float("inf")
        
        if self.eval_steps <= 0:
            self.eval_steps = float("inf")

    @staticmethod
    def _resolve_driver_path(path: str) -> str:
        return os.path.abspath(os.path.expanduser(path))
