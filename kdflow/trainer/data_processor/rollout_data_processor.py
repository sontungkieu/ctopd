import json
import os
from typing import Any, Dict, List, Optional

import torch

from kdflow.datasets.utils import get_tokenizer_or_processor
from kdflow.utils.utils import zero_pad_sequences


class RolloutDataProcessor:
    """Convert raw rollout outputs into training micro-batches."""

    CONDITIONAL_EOS_ALGORITHMS = frozenset(
        {
            "byte_align_ctkd",
            "h1b_span",
            "topk_byte_align_ctkd",
        }
    )

    def __init__(self, strategy, is_same_tokenizer: bool):
        self.args = strategy.args
        self.is_same_tokenizer = is_same_tokenizer
        self.image_key = getattr(self.args.data, "image_key", None)

        self.student_processor = get_tokenizer_or_processor(
            self.args.model.student_name_or_path,
            need_processor=self.image_key is not None,
        )
        self.teacher_processors = {}
        if self.args.model.teacher_name_or_path and not self.is_same_tokenizer:
            self.teacher_processors["default"] = get_tokenizer_or_processor(
                self.args.model.teacher_name_or_path,
                need_processor=self.image_key is not None,
            )

    def process(
        self,
        stu_prompts: List[str],
        tea_prompts: List[str],
        outputs: List[Dict[str, Any]],
        labels: List[Any],
        sampling_params: Dict[str, Any],
        global_step: int,
        mode: str,
        images: Optional[List] = None,
        sample_metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> tuple[List[dict], Dict[str, float]]:
        """Save, tokenize and collate raw rollout outputs."""
        if sample_metadata is not None and len(sample_metadata) != len(outputs):
            raise ValueError("sample_metadata and outputs must have the same length")
        archive_metadata=sample_metadata
        if self.args.kd.kd_algorithm=='h1b_span':
            archive_metadata=[dict(m or {},tea_prompt=t,output_ids=o.get('output_ids',[]))
                              for m,t,o in zip(sample_metadata or [{} for _ in outputs],tea_prompts,outputs)]
        self._save_rollout_data(
            stu_prompts,
            outputs,
            labels,
            global_step,
            mode,
            sample_metadata=archive_metadata,
        )

        kd_algorithm = self.args.kd.kd_algorithm
        uses_conditional_eos = kd_algorithm in self.CONDITIONAL_EOS_ALGORITHMS
        truncated_flags = (
            [
                self._is_length_truncated(output, sampling_params)
                for output in outputs
            ]
            if uses_conditional_eos
            else [False] * len(outputs)
        )

        sample_list = [
            self._build_rollout_sample(
                stu_prompt=stu_prompts[index],
                tea_prompt=tea_prompts[index],
                output=output,
                label=labels[index],
                images=images[index] if images and images[index] else None,
                append_eos=(
                    not truncated_flags[index] if uses_conditional_eos else True
                ),
                metadata=(
                    sample_metadata[index] if sample_metadata is not None else None
                ),
            )
            for index, output in enumerate(outputs)
        ]
        rollout_metrics = self._compute_rollout_metrics(
            sample_list, sampling_params, mode
        )

        if (
            mode == "train"
            and self.args.rollout.print_rollout_sample
            and sample_list
        ):
            print(sample_list[0]["stu_prompts"][0] + sample_list[0]["stu_responses"][0])

        micro_batches = self._collate_micro_batches(
            sample_list, self.args.train.micro_train_batch_size
        )
        return micro_batches, rollout_metrics

    def _save_rollout_data(
        self,
        prompts: List[str],
        outputs: List[Dict[str, Any]],
        labels: List[Any],
        global_step: int,
        mode: str,
        sample_metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        rollout_dir = os.path.join(self.args.train.save_path, "rollout_data")
        if mode == "eval":
            rollout_dir = os.path.join(rollout_dir, "val")
        os.makedirs(rollout_dir, exist_ok=True)

        rollout_path = os.path.join(rollout_dir, f"{global_step}.jsonl")
        with open(rollout_path, "w") as file:
            metadata_rows = sample_metadata or [{} for _ in outputs]
            for prompt, output, label, metadata in zip(
                prompts, outputs, labels, metadata_rows
            ):
                rollout_record = {"prompt": prompt, "output": output["text"]}
                rollout_record.update(metadata)
                finish_reason = self._extract_finish_reason(output)
                if finish_reason is not None:
                    rollout_record["finish_reason"] = finish_reason
                if "reward_result" in output:
                    rollout_record["reward_result"] = output["reward_result"]
                if label is not None and label != "":
                    rollout_record["label"] = label
                file.write(json.dumps(rollout_record, ensure_ascii=False) + "\n")

    @staticmethod
    def _extract_finish_reason(output: Dict[str, Any]):
        meta_info = output.get("meta_info") or {}
        finish_reason = output.get("finish_reason")
        return (
            finish_reason
            if finish_reason is not None
            else meta_info.get("finish_reason")
        )

    @classmethod
    def _is_length_truncated(
        cls,
        output: Dict[str, Any],
        sampling_params: Dict[str, Any],
    ) -> bool:
        finish_reason = cls._extract_finish_reason(output)
        if isinstance(finish_reason, dict):
            reason_type = (
                finish_reason.get("type")
                or finish_reason.get("reason")
                or finish_reason.get("finish_reason")
            )
        else:
            reason_type = finish_reason
        if isinstance(reason_type, str):
            normalized_reason = reason_type.lower().replace("-", "_")
            if normalized_reason in {
                "length",
                "max_length",
                "max_tokens",
                "max_new_tokens",
            }:
                return True

        max_new_tokens = sampling_params.get("max_new_tokens")
        return (
            finish_reason is None
            and max_new_tokens is not None
            and len(output.get("output_ids", ())) >= max_new_tokens
        )

    @staticmethod
    def _compute_rollout_metrics(
        sample_list: List[Dict[str, Any]],
        sampling_params: Dict[str, Any],
        mode: str,
    ) -> Dict[str, float]:
        length_names = ("prompt_length", "response_length", "total_length")
        length_values = {
            name: [sample.pop(name).item() for sample in sample_list]
            for name in length_names
        }
        metric_prefix = "rollout" if mode == "train" else "eval"
        rollout_metrics = {}
        for name, values in length_values.items():
            rollout_metrics[f"{metric_prefix}/{name}/mean"] = sum(values) / len(values)
            rollout_metrics[f"{metric_prefix}/{name}/max"] = max(values)

        max_response_length = sampling_params["max_new_tokens"]
        response_lengths = length_values["response_length"]
        rollout_metrics[f"{metric_prefix}/response_clip_ratio"] = sum(
            length >= max_response_length for length in response_lengths
        ) / len(response_lengths)
        return rollout_metrics

    @staticmethod
    def _collate_values(values: list):
        value = values[0]
        if isinstance(value, torch.Tensor):
            return zero_pad_sequences(values, side="right", value=0)
        if isinstance(value, list):
            return sum(values, [])
        if value is None:
            return None
        return values

    def _collate_micro_batches(
        self, sample_list: List[Dict], batch_size: int
    ) -> List[Dict]:
        """Collate single samples into micro-batches."""
        micro_batch_list = []
        for index in range(0, len(sample_list), batch_size):
            batch_samples = sample_list[index : index + batch_size]
            micro_batch = {
                key: self._collate_values([sample[key] for sample in batch_samples])
                for key in batch_samples[0]
            }
            micro_batch_list.append(micro_batch)
        return micro_batch_list

    def _tokenize_sample(
        self,
        prompt: str,
        response: str,
        processor,
        prefix: str,
        images=None,
        append_eos: bool = True,
    ) -> Dict[str, Any]:
        """Tokenize prompt and response for a single sample."""
        tokenizer = getattr(processor, "tokenizer", processor)
        response_tokens = tokenizer(
            response, return_tensors="pt", add_special_tokens=False
        )
        response_length = response_tokens["input_ids"].shape[1]

        full_input = {"text": prompt + response}
        if images:
            full_input["images"] = images
        full_tokens = processor(
            **full_input, return_tensors="pt", add_special_tokens=False
        )
        prompt_length = full_tokens["input_ids"].shape[1] - response_length

        if append_eos:
            eos_token_id = tokenizer.eos_token_id
            input_ids = torch.cat(
                [
                    full_tokens["input_ids"][0],
                    full_tokens["input_ids"][0].new_tensor([eos_token_id]),
                ]
            )
            attention_mask = torch.cat(
                [
                    full_tokens["attention_mask"][0],
                    full_tokens["attention_mask"][0].new_ones(1),
                ]
            )
        else:
            input_ids = full_tokens["input_ids"][0]
            attention_mask = full_tokens["attention_mask"][0]

        eos_length = int(append_eos)
        loss_mask = torch.tensor(
            [False] * (prompt_length - 1)
            + [True] * (response_length + eos_length)
            + [False],
            device=input_ids.device,
        )

        result = {
            f"{prefix}_input_ids": input_ids,
            f"{prefix}_attn_mask": attention_mask,
            f"{prefix}_loss_mask": loss_mask,
            f"_{prefix}_prompt_length": prompt_length,
        }
        multi_modal_inputs = {
            key: torch.as_tensor(value)
            for key, value in full_tokens.items()
            if key not in ("input_ids", "attention_mask", "mm_token_type_ids")
        }
        if multi_modal_inputs:
            result[f"_{prefix}_multi_modal_inputs"] = multi_modal_inputs
        return result

    def _get_teacher_processor(self):
        """Get the configured teacher processor."""
        if "default" in self.teacher_processors:
            return self.teacher_processors["default"]
        return self.student_processor

    def _build_rollout_sample(
        self,
        stu_prompt: str,
        tea_prompt: str,
        output: Dict[str, Any],
        label: Any,
        images=None,
        append_eos: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build a rollout sample with student and teacher tokenizations."""
        response_ids = output["output_ids"]
        response_text = output["text"]

        stu_tokens = self._tokenize_sample(
            stu_prompt,
            response_text,
            self.student_processor,
            "stu",
            images=images,
            append_eos=append_eos,
        )

        if not self.is_same_tokenizer or tea_prompt != stu_prompt:
            teacher_processor = self._get_teacher_processor()
            tea_tokens = self._tokenize_sample(
                tea_prompt,
                response_text,
                teacher_processor,
                "tea",
                images=images,
                append_eos=append_eos,
            )
        else:
            tea_tokens = {
                "tea_input_ids": stu_tokens["stu_input_ids"].clone(),
                "tea_attn_mask": stu_tokens["stu_attn_mask"].clone(),
                "tea_loss_mask": stu_tokens["stu_loss_mask"].clone(),
            }

        prompt_length = stu_tokens["_stu_prompt_length"]
        response_length = len(response_ids)
        total_length = stu_tokens["stu_attn_mask"].sum().item()

        teacher_processor = self._get_teacher_processor()
        tokenizer = getattr(teacher_processor, "tokenizer", teacher_processor)
        if images:
            feed_ids = tokenizer(
                tea_prompt + response_text, add_special_tokens=False
            )["input_ids"]
            tea_feed_input_ids = (
                feed_ids + [tokenizer.eos_token_id] if append_eos else feed_ids
            )
        else:
            tea_feed_input_ids = tea_tokens["tea_input_ids"].tolist()

        sample = {
            **{key: value for key, value in tea_tokens.items() if not key.startswith("_")},
            **{key: value for key, value in stu_tokens.items() if not key.startswith("_")},
            "tea_feed_input_ids": [tea_feed_input_ids],
            "rollout_log_probs": None,
            "stu_prompts": [stu_prompt],
            "stu_responses": [response_text],
            "tea_prompts": [tea_prompt],
            "labels": [label],
            "prompt_length": torch.FloatTensor([[prompt_length]]),
            "response_length": torch.FloatTensor([[response_length]]),
            "total_length": torch.FloatTensor([[total_length]]),
        }
        stu_multi_modal_inputs = stu_tokens.get("_stu_multi_modal_inputs")
        if self.args.kd.kd_algorithm=='h1b_span':
            tok=getattr(self.student_processor,'tokenizer',self.student_processor)
            encoded=tok(response_text,add_special_tokens=False)['input_ids']
            sample['stu_reencode_match']=torch.FloatTensor([[float(list(encoded)==list(response_ids))]])
        if stu_multi_modal_inputs is not None:
            sample["stu_multi_modal_inputs"] = [stu_multi_modal_inputs]
        if images:
            sample["images"] = [images]
        if metadata:
            reserved = set(sample) & set(metadata)
            if reserved:
                raise ValueError(
                    f"Rollout metadata uses reserved sample keys: {sorted(reserved)}"
                )
            sample.update(metadata)
        return sample
