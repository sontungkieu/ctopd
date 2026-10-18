"""Standalone ALFWorld evaluation with no teacher or optimizer actors."""

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class AlfWorldEvaluator:
    """Evaluate a rollout-only model on a fixed number of ALFWorld episodes."""

    def __init__(self, args, rollout_manager):
        self.args = args
        self.rollout_manager = rollout_manager
        self.num_episodes = args.rollout.alfworld_eval_episodes
        self.batch_size = args.rollout.rollout_batch_size
        self.output_dir = args.train.save_path

        if self.num_episodes <= 0:
            raise ValueError("alfworld_eval_episodes must be positive")
        if self.num_episodes % self.batch_size:
            raise ValueError(
                "alfworld_eval_episodes must be divisible by rollout_batch_size "
                f"(got {self.num_episodes} and {self.batch_size})"
            )

    @staticmethod
    def _trajectory_row(record: Dict[str, Any], batch_index: int) -> Dict[str, Any]:
        output = record.get("output") or {}
        metadata = dict(record.get("metadata") or {})
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
        row.update(metadata)
        return row

    def evaluate(self) -> Dict[str, Any]:
        os.makedirs(self.output_dir, exist_ok=True)
        trajectory_path = os.path.join(self.output_dir, "trajectories.jsonl")
        summary_path = os.path.join(self.output_dir, "summary.json")
        num_batches = self.num_episodes // self.batch_size
        sampling_params = {
            "max_new_tokens": self.args.rollout.generate_max_len,
            "temperature": self.args.rollout.temperature,
            "top_p": self.args.rollout.top_p,
        }

        totals = {
            "episodes": 0.0,
            "episodes_succeeded": 0.0,
            "turns": 0.0,
            "turns_max": 0.0,
            "invalid_actions": 0.0,
            "environment_failures": 0.0,
            "context_truncated": 0.0,
        }
        started_at = time.time()

        logger.info(
            "Starting standalone ALFWorld evaluation: model=%s split=%s "
            "episodes=%d batch_size=%d",
            self.args.model.student_name_or_path,
            self.args.rollout.alfworld_split,
            self.num_episodes,
            self.batch_size,
        )
        with open(trajectory_path, "w", encoding="utf-8") as trajectory_file:
            for batch_index in range(1, num_batches + 1):
                prompt_batch: List[Dict[str, int]] = [
                    {"episode_slot": slot} for slot in range(self.batch_size)
                ]
                records, metrics = self.rollout_manager.collect_episodes(
                    prompt_batch=prompt_batch,
                    global_step=batch_index,
                    mode="eval",
                    **sampling_params,
                )
                for record in records:
                    row = self._trajectory_row(record, batch_index)
                    trajectory_file.write(
                        json.dumps(row, ensure_ascii=False) + "\n"
                    )
                trajectory_file.flush()

                prefix = "eval/alfworld"
                totals["episodes"] += metrics.get(f"{prefix}/episodes", 0.0)
                totals["episodes_succeeded"] += metrics.get(
                    f"{prefix}/episodes_succeeded", 0.0
                )
                totals["turns"] += metrics.get(f"{prefix}/turns/total", 0.0)
                totals["turns_max"] = max(
                    totals["turns_max"],
                    metrics.get(f"{prefix}/turns/max", 0.0),
                )
                totals["invalid_actions"] += metrics.get(
                    f"{prefix}/invalid_action_count", 0.0
                )
                totals["environment_failures"] += metrics.get(
                    f"{prefix}/environment_failure_count", 0.0
                )
                totals["context_truncated"] += metrics.get(
                    f"{prefix}/context_truncated_count", 0.0
                )
                logger.info(
                    "ALFWorld evaluation batch %d/%d complete "
                    "(running success rate %.4f)",
                    batch_index,
                    num_batches,
                    totals["episodes_succeeded"] / max(totals["episodes"], 1.0),
                )

        episodes = totals["episodes"]
        turns = totals["turns"]
        summary = {
            "model": self.args.model.student_name_or_path,
            "split": self.args.rollout.alfworld_split,
            "episodes": int(episodes),
            "episodes_succeeded": int(totals["episodes_succeeded"]),
            "success_rate": totals["episodes_succeeded"] / max(episodes, 1.0),
            "turns": {
                "total": int(turns),
                "mean": turns / max(episodes, 1.0),
                "max": int(totals["turns_max"]),
            },
            "invalid_actions": {
                "count": int(totals["invalid_actions"]),
                "ratio": totals["invalid_actions"] / max(turns, 1.0),
            },
            "environment_failures": {
                "count": int(totals["environment_failures"]),
                "ratio": totals["environment_failures"] / max(turns, 1.0),
            },
            "context_truncated": {
                "count": int(totals["context_truncated"]),
                "ratio": totals["context_truncated"] / max(episodes, 1.0),
            },
            "max_turns_per_episode": self.args.rollout.alfworld_max_turns,
            "sampling": sampling_params,
            "seed": self.args.train.seed,
            "elapsed_seconds": time.time() - started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "trajectories_path": os.path.abspath(trajectory_path),
        }
        with open(summary_path, "w", encoding="utf-8") as summary_file:
            json.dump(summary, summary_file, ensure_ascii=False, indent=2)
            summary_file.write("\n")

        logger.info(
            "ALFWorld evaluation complete: success_rate=%.4f summary=%s",
            summary["success_rate"],
            summary_path,
        )
        return summary
