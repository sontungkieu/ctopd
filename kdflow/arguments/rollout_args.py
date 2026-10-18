from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RolloutArguments:
    """ Arguments for rollout (on-policy distillation)."""
    
    rollout_num_engines: int = field(
        default=0,
        metadata={"help": "The number of engines for rollout."}
    )
    rollout_tp_size: int = field(
        default=1,
        metadata={"help": "Tensor parallel size for each rollout engine."}
    )
    rollout_engine_concurrency: int = field(
        default=512,
        metadata={"help": "Maximum concurrent requests per rollout engine."}
    )
    rollout_enable_sleep: bool = field(
        default=False,
        metadata={"help": "Enable sleep mode for rollout engines."}
    )
    rollout_mem_fraction_static: float = field(
        default=0.6,
        metadata={"help": "GPU memory utilization for each rollout engine."}
    )
    top_p: float = field(
        default=1.0,
        metadata={"help": "Top-p sampling for rollout."}
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature for rollout."}
    )
    n_samples_per_prompt: int = field(
        default=1,
        metadata={"help": "Sample n responses per prompt."}
    )
    rollout_batch_size: int = field(
        default=32,
        metadata={"help": "Number of prompts for each rollout."}
    )
    generate_max_len: int = field(
        default=2048,
        metadata={"help": "Max generation tokens during rollout."}
    )
    print_rollout_sample: bool = field(
        default=False,
        metadata={"help": "Whether to print a rollout sample after each rollout."}
    )
    alfworld_config: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to an ALFWorld YAML config. When set, on-policy KD "
                    "collects multi-turn student trajectories from AlfredTWEnv."
        },
    )
    alfworld_protocol: str = field(
        default="kdflow",
        metadata={
            "help": "ALFWorld training rollout protocol: 'kdflow' keeps the "
                    "original Thought/Action chat history, while 'verl_agent' "
                    "uses the migrated single-prompt <think>/<action> protocol."
        },
    )
    alfworld_split: str = field(
        default="train",
        metadata={
            "help": "ALFWorld split: train, eval_in_distribution, or "
                    "eval_out_of_distribution."
        },
    )
    alfworld_max_turns: int = field(
        default=30,
        metadata={"help": "Maximum assistant/environment turns per ALFWorld episode."},
    )
    alfworld_episodes_per_epoch: int = field(
        default=1024,
        metadata={"help": "Number of ALFWorld episodes sampled per training epoch."},
    )
    alfworld_eval_episodes: int = field(
        default=128,
        metadata={
            "help": "Number of episodes for standalone ALFWorld evaluation."
        },
    )
    alfworld_eval_protocol: str = field(
        default="kdflow",
        metadata={
            "help": "Standalone ALFWorld evaluation protocol. Use 'kdflow' "
                    "for the original KDFlow prompt or 'verl_agent' for the "
                    "protocol migrated from langfengq/verl-agent."
        },
    )
    alfworld_eval_seeds: str = field(
        default="123,456,789",
        metadata={
            "help": "Comma-separated script-level seeds used by the "
                    "verl-agent-compatible evaluator."
        },
    )
    alfworld_eval_seed_offset: int = field(
        default=1000,
        metadata={
            "help": "Offset added to each script-level seed for the validation "
                    "environment. verl-agent uses 1000."
        },
    )
    alfworld_eval_resume: bool = field(
        default=True,
        metadata={
            "help": "Reuse completed, configuration-compatible per-seed "
                    "ALFWorld evaluation results. Set False to force rerunning "
                    "every seed."
        },
    )
    alfworld_history_length: int = field(
        default=4,
        metadata={
            "help": "Number of recent observation/action pairs included by the "
                    "verl-agent ALFWorld prompt."
        },
    )
    alfworld_admissible_limit: int = field(
        default=30,
        metadata={
            "help": "Maximum number of ALFWorld admissible actions shown per turn; "
                    "use 0 to show all actions."
        },
    )
