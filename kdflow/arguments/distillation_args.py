import logging
from dataclasses import dataclass, field
import math

from kdflow.mp_opd_modes import MP_OPD_ALGORITHMS

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
            "choices": ["simple_ctkd", "byte_align_ctkd", "topk_byte_align_ctkd", "simct", "simct_paper", "h1b_span", *MP_OPD_ALGORITHMS],
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

    simct_gh_mask_threshold: float = field(default=0.0,
        metadata={'help': 'Author SimCT optional G(h) mask; 0 disables it as in released launchers.'})

    # Opt-in MP-OPD namespace. None of these fields changes a Strict mode.
    mp_opd_fixed_span_length: int = field(default=2)
    mp_opd_random_min_length: int = field(default=1)
    mp_opd_random_max_length: int = field(default=2)
    mp_opd_random_seed: int = field(default=42)
    mp_opd_chunk_source: str = field(default='xtoken', metadata={'choices': ['xtoken', 'run']})
    mp_opd_chunk_run_length: int = field(default=2)
    mp_opd_chunk_straddle: str = field(default='singleton', metadata={'choices': ['singleton', 'majority']})
    mp_opd_alignment_max_comb_len: int = field(default=4)
    mp_opd_grass_geometry: str = field(default='exact_head', metadata={'choices': ['exact_head', 'diag']})
    mp_opd_grass_sigma_rho: float = field(default=0.99)
    mp_opd_grass_sigma_min_pairs: int = field(default=8)
    mp_opd_align_eps_h: float = field(default=1e-12)
    mp_opd_align_new_min_descent_fraction: float = field(default=0.1)
    mp_opd_align_new_eps_g: float = field(default=1e-12)
    mp_opd_trust_eps_g: float = field(default=1e-12)
    mp_opd_trust_vocab_chunk: int = field(default=4096)

    h1b_arm: str = field(default='a', metadata={'choices':['a','b','b_bl','c_str','c']})
    h1b_span_lambda: float = field(default=0.5)
    h1b_log_sample_rate: float = field(default=0.05)
    h1b_dump_every: str = field(default='0,20,60,100')

    def __post_init__(self):
        if self.multi_teacher_config is not None:
            raise ValueError(
                "This release supports one teacher model per run; "
                "use --teacher_name_or_path instead of --multi_teacher_config."
            )
        supported_algorithms = {"simple_ctkd", "byte_align_ctkd", "topk_byte_align_ctkd", "simct", "simct_paper", "h1b_span", *MP_OPD_ALGORITHMS}
        if self.kd_algorithm not in supported_algorithms:
            raise ValueError(
                f"kd_algorithm must be one of {sorted(supported_algorithms)}, "
                f"got {self.kd_algorithm!r}."
            )
        if self.kd_algorithm == 'h1b_span':
            if self.kd_ratio!=1 or self.kd_temperature!=1 or self.kd_loss_fn!='rkl':
                raise ValueError('H1b requires kd_ratio=1, kd_temperature=1, kd_loss_fn=rkl')
            if self.h1b_arm not in {'a','b','b_bl','c_str','c'}:
                raise ValueError('Invalid H1b arm')
            if not math.isfinite(self.h1b_span_lambda) or self.h1b_span_lambda<0:
                raise ValueError('Invalid H1b span lambda')
            if not math.isfinite(self.h1b_log_sample_rate) or not 0<=self.h1b_log_sample_rate<=1:
                raise ValueError('Invalid H1b sample rate')
            try:
                steps=[int(s) for s in self.h1b_dump_every.split(',') if s.strip()]
            except ValueError as error:
                raise ValueError('H1b dump steps must be comma-separated integers') from error
            if any(s<0 for s in steps): raise ValueError('Negative H1b dump step')
        if self.kd_algorithm in {'simct','simct_paper'}:
            if self.kd_ratio != 1 or self.kd_temperature != 1 or self.kd_loss_fn != 'rkl':
                raise ValueError('SimCT baseline requires kd_ratio=1, kd_temperature=1, kd_loss_fn=rkl')
            if not math.isfinite(self.simct_gh_mask_threshold) or self.simct_gh_mask_threshold < 0:
                raise ValueError('SimCT G(h) threshold must be finite and nonnegative')
        if self.kd_algorithm in MP_OPD_ALGORITHMS:
            if self.kd_algorithm.startswith('mp_opd_trust_'):
                if self.mp_opd_grass_geometry != 'exact_head':
                    raise ValueError('TRUST requires exact_head geometry')
                if not math.isfinite(self.mp_opd_trust_eps_g) or self.mp_opd_trust_eps_g < 0 or self.mp_opd_trust_vocab_chunk < 1:
                    raise ValueError('TRUST requires finite nonnegative eps_g and positive vocabulary tile')
            if self.kd_algorithm == 'mp_opd_align_new':
                if self.mp_opd_grass_geometry != 'exact_head':
                    raise ValueError('ALIGN-NEW requires exact_head geometry')
                if not math.isfinite(self.mp_opd_align_new_min_descent_fraction) or not 0 <= self.mp_opd_align_new_min_descent_fraction <= 1:
                    raise ValueError('ALIGN-NEW minimum descent fraction must be in [0,1]')
                if not math.isfinite(self.mp_opd_align_new_eps_g) or self.mp_opd_align_new_eps_g < 0:
                    raise ValueError('ALIGN-NEW eps_g must be finite and nonnegative')
            if self.kd_ratio != 1.0 or self.kd_temperature != 1.0 or self.kd_loss_fn != 'rkl':
                raise ValueError('MP-OPD requires kd_ratio=1, kd_temperature=1, kd_loss_fn=rkl')
            if self.mp_opd_fixed_span_length < 1 or not 1 <= self.mp_opd_random_min_length <= self.mp_opd_random_max_length:
                raise ValueError('MP-OPD group lengths must be positive and random min <= max')
            if self.mp_opd_chunk_source not in {'xtoken', 'run'} or self.mp_opd_chunk_straddle not in {'singleton', 'majority'}:
                raise ValueError('Unsupported MP-OPD chunk source or straddling policy')
            if self.mp_opd_chunk_run_length < 1 or self.mp_opd_alignment_max_comb_len < 1:
                raise ValueError('MP-OPD chunk/alignment lengths must be positive')
            if self.mp_opd_grass_geometry not in {'exact_head', 'diag'}:
                raise ValueError('Unsupported MP-OPD head geometry')
            if not 0 <= self.mp_opd_grass_sigma_rho < 1 or self.mp_opd_grass_sigma_min_pairs < 1:
                raise ValueError('Invalid GRASS noise estimator parameters')
            if not math.isfinite(self.mp_opd_align_eps_h) or self.mp_opd_align_eps_h <= 0:
                raise ValueError('ALIGN eps_h must be finite and positive')
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
