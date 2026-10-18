import os
from dataclasses import dataclass, field

from transformers import HfArgumentParser

from kdflow.arguments.data_args import DataArguments
from kdflow.arguments.model_args import ModelArguments
from kdflow.arguments.training_args import TrainingArguments
from kdflow.arguments.fsdp_args import FSDPArguments
from kdflow.arguments.distillation_args import DistillationArguments
from kdflow.arguments.rollout_args import RolloutArguments
from kdflow.arguments.logging_args import LoggingArguments
from kdflow.utils.logging_utils import init_logger

logger = init_logger(__name__)

@dataclass
class AllArguments:
    data: DataArguments = field(default_factory=DataArguments)
    model: ModelArguments = field(default_factory=ModelArguments)
    train: TrainingArguments = field(default_factory=TrainingArguments)
    fsdp: FSDPArguments = field(default_factory=FSDPArguments)
    kd: DistillationArguments = field(default_factory=DistillationArguments)
    rollout: RolloutArguments = field(default_factory=RolloutArguments)
    log: LoggingArguments = field(default_factory=LoggingArguments)
    

def is_linear_attention(model_path: str) -> bool:
    if not model_path:
        return False
    try:
        from transformers import AutoConfig
        model_type = AutoConfig.from_pretrained(model_path, trust_remote_code=True).model_type or ""
    except Exception:
        return False
    return any(k in model_type.lower() for k in ("qwen3_next", "qwen3_5"))

def init_args(scenario: str = "on_policy_kd"):
    parser = HfArgumentParser((
        DataArguments,
        ModelArguments,
        TrainingArguments,
        FSDPArguments,
        DistillationArguments,
        RolloutArguments,
        LoggingArguments
    ))
    (
        data_args, 
        model_args, 
        train_args, 
        fsdp_args,
        kd_args, 
        rollout_args, 
        log_args
    ) = parser.parse_args_into_dataclasses()

    args = AllArguments(
        data=data_args,
        model=model_args,
        train=train_args,
        fsdp=fsdp_args,
        kd=kd_args,
        rollout=rollout_args,
        log=log_args
    )

    if scenario != "on_policy_kd" and args.data.custom_eval_fn is not None:
        logger.warning(
            f"`--custom_eval_fn` is only supported in the `on_policy_kd` scenario "
            f"and will be ignored for `{scenario}`."
        )
    if scenario != "on_policy_kd" and args.data.eval_math_dataset_path is not None:
        logger.warning(
            f"`--eval_math_dataset_path` is only supported in the `on_policy_kd` "
            f"scenario and will be ignored for `{scenario}`."
        )

    if args.data.input_template and "{}" not in args.data.input_template:
        logger.warning("{} not in args.data.input_template, set to None")
        args.data.input_template = None

    if args.data.input_template and "\\n" in args.data.input_template:
        logger.warning(
            "input_template contains \\n characters instead of newline. "
            "You likely want to pass $'\\n' in Bash or \"`n\" in PowerShell."
        )

    if args.data.packing_samples:
        if "flash_attention" not in args.model.attn_implementation:
            logger.warning(
                "Please use --attn_implementation with flash_attention to accelerate when --packing_samples is enabled."
            )
            args.model.attn_implementation = "flash_attention_2"

        if is_linear_attention(args.model.student_name_or_path):
            logger.warning(
                f"--packing_samples is not yet compatible with linear-attention models "
                f"(got {args.model.student_name_or_path}). Auto-disabling --packing_samples."
            )
            args.data.packing_samples = False

    total_gpus = args.train.num_nodes * args.train.num_gpus_per_node

    if scenario == "alfworld_eval":
        if not args.model.student_name_or_path:
            raise ValueError(
                "Standalone ALFWorld evaluation requires "
                "`--student_name_or_path`."
            )
        if not args.rollout.alfworld_config:
            raise ValueError(
                "Standalone ALFWorld evaluation requires `--alfworld_config`."
            )
        valid_protocols = {"kdflow", "verl_agent"}
        if args.rollout.alfworld_eval_protocol not in valid_protocols:
            raise ValueError(
                "`--alfworld_eval_protocol` must be one of "
                f"{sorted(valid_protocols)}."
            )
        if args.rollout.alfworld_eval_protocol == "verl_agent":
            try:
                seeds = [
                    int(value.strip())
                    for value in args.rollout.alfworld_eval_seeds.split(",")
                    if value.strip()
                ]
            except ValueError as error:
                raise ValueError(
                    "`--alfworld_eval_seeds` must be a comma-separated list "
                    "of integers."
                ) from error
            if not seeds:
                raise ValueError(
                    "`--alfworld_eval_seeds` must contain at least one seed."
                )
            if args.rollout.alfworld_history_length < 0:
                raise ValueError(
                    "`--alfworld_history_length` must be non-negative."
                )

    if scenario in ("on_policy_kd", "alfworld_eval"):
        if total_gpus <= 0:
            raise ValueError(
                "`--num_nodes * --num_gpus_per_node` must be positive."
            )
        if args.rollout.rollout_tp_size <= 0:
            raise ValueError("`--rollout_tp_size` must be positive.")
        if args.rollout.rollout_batch_size <= 0:
            raise ValueError("`--rollout_batch_size` must be positive.")
        if args.rollout.rollout_engine_concurrency <= 0:
            raise ValueError(
                "`--rollout_engine_concurrency` must be positive."
            )
        if args.rollout.alfworld_config:
            valid_alfworld_protocols = {"kdflow", "verl_agent"}
            if args.rollout.alfworld_protocol not in valid_alfworld_protocols:
                raise ValueError(
                    "`--alfworld_protocol` must be one of "
                    f"{sorted(valid_alfworld_protocols)}."
                )
            args.rollout.alfworld_config = os.path.abspath(
                os.path.expanduser(args.rollout.alfworld_config)
            )
            if not os.path.isfile(args.rollout.alfworld_config):
                raise ValueError(
                    "`--alfworld_config` must point to an existing YAML file, "
                    f"got {args.rollout.alfworld_config!r}."
                )
            valid_alfworld_splits = {
                "train",
                "eval_in_distribution",
                "eval_out_of_distribution",
            }
            if args.rollout.alfworld_split not in valid_alfworld_splits:
                raise ValueError(
                    "`--alfworld_split` must be one of "
                    f"{sorted(valid_alfworld_splits)}."
                )
            if args.rollout.alfworld_max_turns <= 0:
                raise ValueError("`--alfworld_max_turns` must be positive.")
            if args.rollout.alfworld_episodes_per_epoch <= 0:
                raise ValueError(
                    "`--alfworld_episodes_per_epoch` must be positive."
                )
            if args.rollout.alfworld_eval_episodes <= 0:
                raise ValueError("`--alfworld_eval_episodes` must be positive.")
            if args.rollout.alfworld_admissible_limit < 0:
                raise ValueError(
                    "`--alfworld_admissible_limit` must be non-negative."
                )
            if args.rollout.n_samples_per_prompt != 1:
                raise ValueError(
                    "ALFWorld multi-turn rollout requires "
                    "`--n_samples_per_prompt 1`; the environment already "
                    "samples one independent game per batch slot."
                )
            if scenario == "on_policy_kd" and (
                args.rollout.alfworld_episodes_per_epoch
                < args.rollout.rollout_batch_size
            ):
                raise ValueError(
                    "`--alfworld_episodes_per_epoch` must be at least "
                    "`--rollout_batch_size`."
                )
            remainder = (
                args.rollout.alfworld_episodes_per_epoch
                % args.rollout.rollout_batch_size
            )
            if scenario == "on_policy_kd" and remainder:
                logger.warning(
                    "ALFWorld drops %d episode slots per epoch because "
                    "alfworld_episodes_per_epoch is not divisible by "
                    "rollout_batch_size.",
                    remainder,
                )
            if scenario == "on_policy_kd" and args.data.eval_dataset_path:
                logger.warning(
                    "`--eval_dataset_path` is ignored for ALFWorld multi-turn "
                    "training; use an ALFWorld evaluation split in a separate run."
                )

            if (
                scenario == "alfworld_eval"
                and args.rollout.alfworld_eval_episodes
                % args.rollout.rollout_batch_size
            ):
                raise ValueError(
                    "`--alfworld_eval_episodes` must be divisible by "
                    "`--rollout_batch_size`."
                )

        if scenario == "on_policy_kd":
            if args.data.eval_math_n_samples <= 0:
                raise ValueError("`--eval_math_n_samples` must be positive.")
            if args.data.eval_math_max_concurrent <= 0:
                raise ValueError("`--eval_math_max_concurrent` must be positive.")
            if args.data.eval_math_max_tokens <= 0:
                raise ValueError("`--eval_math_max_tokens` must be positive.")
            if args.data.eval_math_timeout < 0:
                raise ValueError("`--eval_math_timeout` must be non-negative.")

        if total_gpus % args.rollout.rollout_tp_size != 0:
            raise ValueError(
                f"Total GPUs ({total_gpus}) must be divisible by rollout_tp_size ({args.rollout.rollout_tp_size})."
            )
            
        expected_num_engines = total_gpus // args.rollout.rollout_tp_size
        if args.rollout.rollout_num_engines != expected_num_engines:
            logger.warning(
                f"Auto-adjusting rollout_num_engines from {args.rollout.rollout_num_engines} to {expected_num_engines} "
                f"to match total GPUs ({total_gpus}). "
                f"(rollout_tp_size={args.rollout.rollout_tp_size} * rollout_num_engines={expected_num_engines} = {total_gpus})"
            )
            args.rollout.rollout_num_engines = expected_num_engines
            
        if args.data.max_len < args.data.prompt_max_len + args.rollout.generate_max_len:
            args.data.max_len = args.data.prompt_max_len + args.rollout.generate_max_len
            logger.warning(
                "--max_len is smaller than --prompt_max_len + --generate_max_len. "
                f"Automatically increase --max_len to {args.data.max_len}."
            )
    
    has_teacher = args.model.teacher_name_or_path is not None
    is_kd_scenario = scenario in ("off_policy_kd", "on_policy_kd")
    if is_kd_scenario:
        if not has_teacher:
            raise ValueError(
                "KD training requires `--teacher_name_or_path` to be set."
            )

        teacher_parallel = args.kd.teacher_tp_size * args.kd.teacher_pp_size
        if total_gpus % teacher_parallel != 0:
            raise ValueError(
                f"Total GPUs ({total_gpus}) must be divisible by "
                f"teacher_tp_size * teacher_pp_size ({args.kd.teacher_tp_size} * {args.kd.teacher_pp_size} = {teacher_parallel})."
            )
            
        if args.kd.teacher_ep_size > args.kd.teacher_tp_size:
            logger.warning(
                f"teacher_ep_size ({args.kd.teacher_ep_size}) must not be greater than teacher_tp_size ({args.kd.teacher_tp_size}). "
                f"Auto-adjusting teacher_ep_size to {args.kd.teacher_tp_size}."
            )
            args.kd.teacher_ep_size = args.kd.teacher_tp_size
            
        expected_dp = total_gpus // teacher_parallel
        if args.kd.teacher_dp_size != expected_dp:
            logger.warning(
                f"Auto-adjusting teacher_dp_size from {args.kd.teacher_dp_size} to {expected_dp} "
                f"to match total GPUs ({total_gpus}). "
                f"(tp={args.kd.teacher_tp_size} (ep={args.kd.teacher_ep_size}) * pp={args.kd.teacher_pp_size} * dp={expected_dp} = {total_gpus})"
            )
            args.kd.teacher_dp_size = expected_dp

    
    deprecated_sleep_flags = [
        args.train.train_enable_sleep,
        args.kd.teacher_enable_sleep,
        args.rollout.rollout_enable_sleep,
    ]
    if any(deprecated_sleep_flags):
        logger.warning(
            "--train_enable_sleep, --teacher_enable_sleep, --rollout_enable_sleep are deprecated "
            "and will be removed in a future version. Use --enable_sleep instead."
        )
        args.train.enable_sleep = True

    args.scenario = scenario
    return args
