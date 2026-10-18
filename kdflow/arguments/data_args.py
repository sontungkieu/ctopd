from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataArguments:
    """ Arguments for dataset."""

    train_dataset_path: str = field(
        default=None,
        metadata={"help": "Training dataset name or path."}
    )
    train_dataset_probs: str = field(
        default=None,
        metadata={"help": "Sampling probs for multiple datasets."}
    )
    train_split: str = field(
        default="train",
        metadata={"help": "Train split in dataset."}
    )
    eval_dataset_path: str = field(
        default=None,
        metadata={"help": "Evaluation dataset name or path."}
    )
    eval_split: str = field(
        default="eval",
        metadata={"help": "Eval split in dataset."}
    )
    custom_eval_fn: str = field(
        default=None,
        metadata={
            "help": "Optional Python file defining eval_fn(predictions, labels), "
                    "which must return a metric dict (only supported in on-policy KD)."
        }
    )
    eval_math_dataset_path: str = field(
        default=None,
        metadata={
            "help": "Optional math evaluation JSONL path. Each row must contain "
                    "'problem' and 'answer' fields (only supported in on-policy KD)."
        }
    )
    eval_math_n_samples: int = field(
        default=1,
        metadata={"help": "Number of generated answers per math problem for pass@k evaluation."}
    )
    eval_math_max_concurrent: int = field(
        default=256,
        metadata={"help": "Maximum number of concurrent math evaluation requests."}
    )
    eval_math_max_tokens: int = field(
        default=8192,
        metadata={"help": "Maximum number of new tokens generated for math evaluation."}
    )
    eval_math_timeout: int = field(
        default=1800,
        metadata={"help": "Total timeout in seconds for math evaluation generation."}
    )
    input_key: str = field(
        default="messages",
        metadata={"help": "JSON dataset key."}
    )
    image_key: str = field(
        default=None,
        metadata={"help": "Image key in dataset."}
    )
    output_key: str = field(
        default=None,
        metadata={"help": "Output key in dataset."}
    )
    teacher_input_key: str = field(
        default=None,
        metadata={"help": "Input key for teacher prompt. If None, use the same input_key as student. "
                          "Used in self-distillation where teacher receives richer information."}
    )
    label_key: str = field(
        default=None,
        metadata={"help": "Label key in dataset."}
    )
    teacher_routing_key: str = field(
        default="teacher_routing_key",
        metadata={"help": "Route key for teacher model in multi-teacher distillation."}
    )
    input_template: str = field(
        default=None,
    )
    apply_chat_template: bool = field(
        default=True,
        metadata={"help": "Use HF tokenizer chat template."}
    )
    max_len: int = field(
        default=4096
    )
    max_samples: int = field(
        default=1e8
    )
    packing_samples: bool = field(
        default=False,
        metadata={"help": "Packing sequences during training."}
    )
    prompt_max_len: int = field(
        default=2048,
        metadata={"help": "Max prompt length."}
    )
    preprocess_num_workers: int = field(
        default=1,
    )
