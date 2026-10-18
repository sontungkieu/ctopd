import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DistillationArguments:
    """ Arguments for knowledge distillation."""
    
    kd_ratio: float = field(
        default=0.5,
        metadata={"help": "Loss = (1 - kd_ratio) * nll_loss + kd_ratio * kd_loss."}
    )
    kd_temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature for knowledge distillation."}
    )
    kd_algorithm: str = field(
        default="byte_align_ctkd",
        metadata={
            "help": "KD algorithm for each training step.",
            "choices": ["simple_ctkd", "byte_align_ctkd", "topk_byte_align_ctkd"],
        }
    )
    kd_loss_fn: str = field(
        default="kl",
        metadata={"help": "Divergence selection for knowledge distillation, e.g., kl, rkl, js."}
    )
    ctkd_topk: int = field(
        default=128,
        metadata={
            "help": (
                "Student top-k size within the shared vocabulary for "
                "topk_byte_align_ctkd."
            )
        },
    )
    teacher_forward_n_batches: int = field(
        default=1,
        metadata={"help": "Teacher forward N global batches at once for student multi-step training."}
    )
    teacher_enable_sleep: bool = field(
        default=False,
        metadata={"help": "Sleep teacher when not needed."}
    )
    teacher_offload_tags: str = field(
        default="all",
        metadata={"help": "Offload tags for sglang."}
    )
    teacher_quantization: str = field(
        default=None
    )
    teacher_tp_size: int = field(
        default=8,
        metadata={"help": "Tensor parallel size for teacher model."}
    )
    teacher_ep_size: int = field(
        default=1,
        metadata={"help": "Expert parallel size for teacher model (only for MoE models)."}
    )
    teacher_pp_size: int = field(
        default=1,
        metadata={"help": "Pipeline parallel size for teacher model."}
    )
    teacher_dp_size: int = field(
        default=1,
        metadata={"help": "Data parallel size for teacher model."}
    )
    teacher_mem_fraction_static: float = field(
        default=0.4,
        metadata={"help": "Memory fraction for teacher model."}
    )
    teacher_update_freq: int = field(
        default=10,
        metadata={"help": "Weight update frequency for teacher model."}
    )
    use_ema_teacher: bool = field(
        default=False,
        metadata={"help": "Use EMA update for teacher model."}
    )
    teacher_ema_decay: float = field(
        default=0.999,
        metadata={"help": "EMA decay for teacher model."}
    )
    multi_teacher_config: str = field(
        default=None,
        metadata={"help": "Unsupported in this release; use --teacher_name_or_path."},
    )
    # JSD
    jsd_beta: float = field(
        default=0.5,
        metadata={"help": "Beta for Jensen-Shannon Divergence."}
    )
    # Skewed KL/RKL
    skew_lambda: float = field(
        default=0.1,
        metadata={"help": "Lambda for Skewed KL/RKL."}
    )
    # Adaptive KL
    adaptive_alpha: float = field(
        default=0.5,
        metadata={"help": "Alpha for Adaptive KL Divergence."}
    )
    # Hierarchical Ranking Loss
    hrl_topk: int = field(
        default=5,
        metadata={"help": "Top-k Ranking for Hierarchical Ranking Loss."}
    )

    def __post_init__(self):
        if self.multi_teacher_config is not None:
            raise ValueError(
                "This release supports one teacher model per run; "
                "use --teacher_name_or_path instead of --multi_teacher_config."
            )
        supported_algorithms = {"simple_ctkd", "byte_align_ctkd", "topk_byte_align_ctkd"}
        if self.kd_algorithm not in supported_algorithms:
            raise ValueError(
                f"kd_algorithm must be one of {sorted(supported_algorithms)}, "
                f"got {self.kd_algorithm!r}."
            )
        # Validate teacher parallel size settings
        if self.teacher_ep_size > self.teacher_tp_size:
            raise ValueError(
                f"SGLang requires that teacher_ep_size ({self.teacher_ep_size}) must be <= teacher_tp_size ({self.teacher_tp_size}). "
            )
        if self.teacher_tp_size % self.teacher_ep_size != 0:
            raise ValueError(
                f"SGLang requires that teacher_tp_size ({self.teacher_tp_size}) must be divisible by teacher_ep_size ({self.teacher_ep_size})."
            )
        # Validate KD hyperparameters
        if not 0.0 <= self.kd_ratio <= 1.0:
            raise ValueError(f"kd_ratio must be in [0, 1], got {self.kd_ratio}.")
        if self.kd_temperature <= 0:
            raise ValueError(f"kd_temperature must be > 0, got {self.kd_temperature}.")
        if self.ctkd_topk < 2:
            raise ValueError(
                f"ctkd_topk must be at least 2, got {self.ctkd_topk}."
            )
        if not 0.0 < self.teacher_mem_fraction_static <= 1.0:
            raise ValueError(f"teacher_mem_fraction_static must be in (0, 1], got {self.teacher_mem_fraction_static}.")
