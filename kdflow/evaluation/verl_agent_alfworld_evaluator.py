"""Multi-seed result writer for the migrated verl-agent ALFWorld protocol."""

import json
import logging
import os
import statistics
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

from kdflow.evaluation.verl_agent_alfworld_protocol import TASK_TYPES


logger = logging.getLogger(__name__)


class VerlAgentAlfWorldEvaluator:
    """Evaluate one model with verl-agent's three-seed ALFWorld recipe."""

    def __init__(self, args, rollout_manager):
        self.args = args
        self.rollout_manager = rollout_manager
        self.num_episodes = args.rollout.alfworld_eval_episodes
        self.batch_size = args.rollout.rollout_batch_size
        self.output_dir = args.train.save_path
        self.seeds = self.parse_seeds(args.rollout.alfworld_eval_seeds)
        if self.num_episodes <= 0:
            raise ValueError("alfworld_eval_episodes must be positive")
        if self.num_episodes % self.batch_size:
            raise ValueError(
                "alfworld_eval_episodes must be divisible by rollout_batch_size"
            )

    @staticmethod
    def parse_seeds(value: str) -> List[int]:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
        if not seeds:
            raise ValueError("At least one ALFWorld evaluation seed is required.")
        return seeds

    @staticmethod
    def _trajectory_row(record: Dict[str, Any], batch_index: int) -> Dict[str, Any]:
        output = record.get("output") or {}
        row = {
            "batch_index": batch_index,
            "prompt": record.get("stu_prompt", ""),
            "response": output.get("text", ""),
        }
        finish_reason = output.get("finish_reason")
        if finish_reason is None:
            finish_reason = (output.get("meta_info") or {}).get("finish_reason")
        if finish_reason is not None:
            row["finish_reason"] = finish_reason
        row.update(record.get("metadata") or {})
        return row

    def _evaluate_seed(self, script_seed: int) -> Dict[str, Any]:
        if self.rollout_manager is None:
            raise RuntimeError(
                "An ALFWorld rollout manager is required for an unfinished seed."
            )
        seed_dir = os.path.join(self.output_dir, f"seed_{script_seed}")
        os.makedirs(seed_dir, exist_ok=True)
        trajectory_path = os.path.join(seed_dir, "trajectories.jsonl")
        summary_path = os.path.join(seed_dir, "summary.json")
        sampling = {
            "max_new_tokens": self.args.rollout.generate_max_len,
            "temperature": self.args.rollout.temperature,
            "top_p": self.args.rollout.top_p,
        }
        prefix = "eval/alfworld"
        totals = {
            "episodes": 0.0,
            "succeeded": 0.0,
            "turns": 0.0,
            "turns_max": 0.0,
            "format_invalid": 0.0,
            "inadmissible": 0.0,
            "environment_failures": 0.0,
            "prompt_truncated": 0.0,
        }
        task_totals = {
            task: {"episodes": 0.0, "succeeded": 0.0}
            for task in TASK_TYPES + ("unknown",)
        }
        num_batches = self.num_episodes // self.batch_size
        started_at = time.time()

        with open(trajectory_path, "w", encoding="utf-8") as output_file:
            for batch_index in range(num_batches):
                episode_offset = batch_index * self.batch_size
                self.rollout_manager.set_evaluation_batch(
                    script_seed=script_seed,
                    episode_offset=episode_offset,
                )
                prompt_batch: List[Dict[str, int]] = [
                    {"episode_slot": episode_offset + slot}
                    for slot in range(self.batch_size)
                ]
                records, metrics = self.rollout_manager.collect_episodes(
                    prompt_batch=prompt_batch,
                    global_step=batch_index + 1,
                    mode="eval",
                    **sampling,
                )
                for record in records:
                    output_file.write(
                        json.dumps(
                            self._trajectory_row(record, batch_index + 1),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                output_file.flush()

                totals["episodes"] += metrics.get(f"{prefix}/episodes", 0.0)
                totals["succeeded"] += metrics.get(
                    f"{prefix}/episodes_succeeded", 0.0
                )
                totals["turns"] += metrics.get(f"{prefix}/turns/total", 0.0)
                totals["turns_max"] = max(
                    totals["turns_max"],
                    metrics.get(f"{prefix}/turns/max", 0.0),
                )
                totals["format_invalid"] += metrics.get(
                    f"{prefix}/format_invalid_count", 0.0
                )
                totals["inadmissible"] += metrics.get(
                    f"{prefix}/inadmissible_action_count", 0.0
                )
                totals["environment_failures"] += metrics.get(
                    f"{prefix}/environment_failure_count", 0.0
                )
                totals["prompt_truncated"] += metrics.get(
                    f"{prefix}/prompt_truncated_count", 0.0
                )
                for task in task_totals:
                    task_totals[task]["episodes"] += metrics.get(
                        f"{prefix}/task/{task}/episodes", 0.0
                    )
                    task_totals[task]["succeeded"] += metrics.get(
                        f"{prefix}/task/{task}/succeeded", 0.0
                    )
                logger.info(
                    "verl-agent ALFWorld seed=%d batch=%d/%d running_success=%.4f",
                    script_seed,
                    batch_index + 1,
                    num_batches,
                    totals["succeeded"] / max(totals["episodes"], 1.0),
                )

        episodes = totals["episodes"]
        turns = totals["turns"]
        tasks = {}
        for task, values in task_totals.items():
            if not values["episodes"]:
                continue
            tasks[task] = {
                "episodes": int(values["episodes"]),
                "succeeded": int(values["succeeded"]),
                "success_rate": values["succeeded"] / values["episodes"],
            }
        summary = {
            "protocol": "verl_agent",
            "source_commit": "20bd331bdbc9026a5668e11362178e10ab7400c8",
            "model": self.args.model.student_name_or_path,
            "split": self.args.rollout.alfworld_split,
            "script_seed": script_seed,
            "environment_seed_base": (
                script_seed + self.args.rollout.alfworld_eval_seed_offset
            ),
            "episodes": int(episodes),
            "episodes_succeeded": int(totals["succeeded"]),
            "success_rate": totals["succeeded"] / max(episodes, 1.0),
            "turns": {
                "total": int(turns),
                "mean": turns / max(episodes, 1.0),
                "max": int(totals["turns_max"]),
            },
            "format_invalid": {
                "count": int(totals["format_invalid"]),
                "ratio": totals["format_invalid"] / max(turns, 1.0),
            },
            "inadmissible_actions": {
                "count": int(totals["inadmissible"]),
                "ratio": totals["inadmissible"] / max(turns, 1.0),
            },
            "environment_failures": {
                "count": int(totals["environment_failures"]),
                "ratio": totals["environment_failures"] / max(turns, 1.0),
            },
            "prompt_truncated": {
                "count": int(totals["prompt_truncated"]),
                "ratio": totals["prompt_truncated"] / max(turns, 1.0),
            },
            "task_types": tasks,
            "history_length": self.args.rollout.alfworld_history_length,
            "max_steps": self.args.rollout.alfworld_max_turns,
            "sampling": sampling,
            "elapsed_seconds": time.time() - started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "trajectories_path": os.path.abspath(trajectory_path),
        }
        with open(summary_path, "w", encoding="utf-8") as output_file:
            json.dump(summary, output_file, ensure_ascii=False, indent=2)
            output_file.write("\n")
        return summary

    def _load_completed_seed(self, script_seed: int):
        """Return a compatible completed seed summary, otherwise ``None``."""
        seed_dir = os.path.join(self.output_dir, f"seed_{script_seed}")
        summary_path = os.path.join(seed_dir, "summary.json")
        trajectory_path = os.path.join(seed_dir, "trajectories.jsonl")
        if not os.path.isfile(summary_path) or not os.path.isfile(trajectory_path):
            return None
        if os.path.getsize(trajectory_path) <= 0:
            logger.warning(
                "Existing ALFWorld seed=%d trajectory is empty; rerunning.",
                script_seed,
            )
            return None

        try:
            with open(summary_path, "r", encoding="utf-8") as input_file:
                summary = json.load(input_file)
        except (OSError, json.JSONDecodeError) as error:
            logger.warning(
                "Cannot read existing ALFWorld seed=%d summary (%s); rerunning.",
                script_seed,
                error,
            )
            return None

        expected_sampling = {
            "max_new_tokens": self.args.rollout.generate_max_len,
            "temperature": self.args.rollout.temperature,
            "top_p": self.args.rollout.top_p,
        }
        expected = {
            "protocol": "verl_agent",
            "source_commit": "20bd331bdbc9026a5668e11362178e10ab7400c8",
            "model": self.args.model.student_name_or_path,
            "split": self.args.rollout.alfworld_split,
            "script_seed": script_seed,
            "environment_seed_base": (
                script_seed + self.args.rollout.alfworld_eval_seed_offset
            ),
            "episodes": self.num_episodes,
            "history_length": self.args.rollout.alfworld_history_length,
            "max_steps": self.args.rollout.alfworld_max_turns,
            "sampling": expected_sampling,
        }
        mismatches = [
            key for key, value in expected.items() if summary.get(key) != value
        ]
        required_fields = {
            "episodes_succeeded",
            "success_rate",
            "turns",
            "task_types",
        }
        missing_fields = sorted(required_fields - summary.keys())
        if mismatches or missing_fields:
            logger.warning(
                "Existing ALFWorld seed=%d result is incompatible "
                "(mismatched=%s missing=%s); rerunning.",
                script_seed,
                mismatches,
                missing_fields,
            )
            return None
        turns = summary.get("turns") or {}
        if not {"total", "mean", "max"}.issubset(turns):
            logger.warning(
                "Existing ALFWorld seed=%d result has incomplete turn metrics; "
                "rerunning.",
                script_seed,
            )
            return None
        return summary

    def has_pending_seeds(self) -> bool:
        """Whether evaluation must load a model and generate new trajectories."""
        if not getattr(self.args.rollout, "alfworld_eval_resume", True):
            return True
        return any(self._load_completed_seed(seed) is None for seed in self.seeds)

    def evaluate(self) -> Dict[str, Any]:
        os.makedirs(self.output_dir, exist_ok=True)
        started_at = time.time()
        resume_enabled = getattr(
            self.args.rollout, "alfworld_eval_resume", True
        )
        seed_summaries = []
        skipped_existing_seeds = []
        evaluated_seeds = []
        for seed in self.seeds:
            existing = self._load_completed_seed(seed) if resume_enabled else None
            if existing is not None:
                logger.info(
                    "Skipping completed compatible ALFWorld seed=%d result.",
                    seed,
                )
                seed_summaries.append(existing)
                skipped_existing_seeds.append(seed)
                continue
            seed_summaries.append(self._evaluate_seed(seed))
            evaluated_seeds.append(seed)
        rates = [summary["success_rate"] for summary in seed_summaries]
        total_episodes = sum(summary["episodes"] for summary in seed_summaries)
        total_succeeded = sum(
            summary["episodes_succeeded"] for summary in seed_summaries
        )
        total_turns = sum(
            summary["turns"]["total"] for summary in seed_summaries
        )
        seed_turn_means = [
            summary["turns"]["mean"] for summary in seed_summaries
        ]
        max_turns = max(
            (summary["turns"]["max"] for summary in seed_summaries),
            default=0,
        )

        task_names = set().union(
            *(summary["task_types"].keys() for summary in seed_summaries)
        )
        task_summary = {}
        for task in sorted(task_names):
            per_seed_rates = [
                summary["task_types"][task]["success_rate"]
                for summary in seed_summaries
                if task in summary["task_types"]
            ]
            task_episodes = sum(
                summary["task_types"].get(task, {}).get("episodes", 0)
                for summary in seed_summaries
            )
            task_succeeded = sum(
                summary["task_types"].get(task, {}).get("succeeded", 0)
                for summary in seed_summaries
            )
            task_summary[task] = {
                "episodes": task_episodes,
                "succeeded": task_succeeded,
                "micro_success_rate": task_succeeded / max(task_episodes, 1),
                "seed_mean_success_rate": statistics.fmean(per_seed_rates),
                "seed_std_success_rate": (
                    statistics.pstdev(per_seed_rates)
                    if len(per_seed_rates) > 1
                    else 0.0
                ),
            }

        summary = {
            "protocol": "verl_agent",
            "source_commit": "20bd331bdbc9026a5668e11362178e10ab7400c8",
            "model": self.args.model.student_name_or_path,
            "split": self.args.rollout.alfworld_split,
            "seeds": self.seeds,
            "environment_seed_offset": self.args.rollout.alfworld_eval_seed_offset,
            "resume_enabled": resume_enabled,
            "skipped_existing_seeds": skipped_existing_seeds,
            "evaluated_seeds": evaluated_seeds,
            "episodes_per_seed": self.num_episodes,
            "total_episodes": total_episodes,
            "total_succeeded": total_succeeded,
            "micro_success_rate": total_succeeded / max(total_episodes, 1),
            "seed_mean_success_rate": statistics.fmean(rates),
            "seed_std_success_rate": (
                statistics.pstdev(rates) if len(rates) > 1 else 0.0
            ),
            "turns": {
                "total": total_turns,
                "micro_mean": total_turns / max(total_episodes, 1),
                "seed_mean": statistics.fmean(seed_turn_means),
                "seed_std": (
                    statistics.pstdev(seed_turn_means)
                    if len(seed_turn_means) > 1
                    else 0.0
                ),
                "max": max_turns,
            },
            "task_types": task_summary,
            "history_length": self.args.rollout.alfworld_history_length,
            "max_steps": self.args.rollout.alfworld_max_turns,
            "sampling": {
                "max_new_tokens": self.args.rollout.generate_max_len,
                "temperature": self.args.rollout.temperature,
                "top_p": self.args.rollout.top_p,
            },
            "seed_summaries": [
                os.path.join(f"seed_{seed}", "summary.json")
                for seed in self.seeds
            ],
            "elapsed_seconds": time.time() - started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        summary_path = os.path.join(self.output_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as output_file:
            json.dump(summary, output_file, ensure_ascii=False, indent=2)
            output_file.write("\n")
        logger.info(
            "verl-agent ALFWorld evaluation complete: mean=%.4f std=%.4f "
            "turns=%.2f",
            summary["seed_mean_success_rate"],
            summary["seed_std_success_rate"],
            summary["turns"]["micro_mean"],
        )
        return summary
