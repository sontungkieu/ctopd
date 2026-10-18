# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Adapted for KDFlow from langfengq/verl-agent at commit
# 20bd331bdbc9026a5668e11362178e10ab7400c8.

"""ALFWorld rollout protocol compatible with the verl-agent evaluation recipe."""

import copy
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from kdflow.evaluation.verl_agent_alfworld_protocol import (
    ALFWORLD_TEMPLATE_NO_HISTORY,
    ALFWORLD_TEMPLATE_WITH_HISTORY,
    TASK_TYPES,
    project_alfworld_action,
    task_type_from_gamefile,
)
from kdflow.trainer.rollout_manager import RolloutManager
from kdflow.utils.logging_utils import init_logger


logger = init_logger(__name__)


class _IndependentAlfWorldBatch:
    """A small local equivalent of verl-agent's one-Ray-actor-per-game setup."""

    def __init__(self, config, split: str, batch_size: int, base_seed: int):
        try:
            from alfworld.agents.environment import get_environment
        except ImportError as error:
            raise ImportError(
                "The verl-agent-compatible evaluator requires ALFWorld. Install "
                "it with `pip install -e '.[alfworld]'` and make the downloaded "
                "data available through ALFWORLD_DATA."
            ) from error

        env_type = config.get("env", {}).get("type", "AlfredTWEnv")
        if env_type != "AlfredTWEnv":
            raise ValueError(
                "The migrated evaluator supports text-only AlfredTWEnv, got "
                f"{env_type!r}."
            )
        self.batch_size = batch_size
        self.base_seed = base_seed
        self.episode_offset = 0
        self.envs = []
        for index in range(batch_size):
            worker_config = copy.deepcopy(config)
            worker_seed = base_seed + index
            worker_config.setdefault("general", {})["random_seed"] = worker_seed
            base_env = get_environment(env_type)(worker_config, train_eval=split)
            env = base_env.init_env(batch_size=1)
            seed_method = getattr(env, "seed", None)
            if callable(seed_method):
                seed_method(worker_seed)
            self.envs.append(env)

    def set_episode_range(self, base_seed: int, episode_offset: int) -> None:
        self.base_seed = base_seed
        self.episode_offset = episode_offset

    @staticmethod
    def _scalar(value):
        if isinstance(value, (list, tuple)):
            return value[0] if value else None
        try:
            if getattr(value, "shape", None) == (1,):
                return value[0]
        except Exception:
            pass
        return value

    @classmethod
    def _merge_infos(cls, infos: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        keys = set().union(*(info.keys() for info in infos)) if infos else set()
        return {
            key: [cls._scalar(info.get(key)) for info in infos]
            for key in keys
        }

    def reset(self):
        observations = []
        infos = []
        for index, env in enumerate(self.envs):
            seed_method = getattr(env, "seed", None)
            if callable(seed_method):
                seed_method(self.base_seed + self.episode_offset + index)
            observation, info = env.reset()
            observations.append(self._scalar(observation))
            infos.append(info)
        return observations, self._merge_infos(infos)

    def step(self, actions: Sequence[str]):
        observations = []
        rewards = []
        dones = []
        infos = []
        for env, action in zip(self.envs, actions):
            observation, reward, done, info = env.step([action])
            observations.append(self._scalar(observation))
            rewards.append(self._scalar(reward))
            dones.append(bool(self._scalar(done)))
            infos.append(info)
        return observations, rewards, dones, self._merge_infos(infos)

    def close(self) -> None:
        for env in self.envs:
            close = getattr(env, "close", None)
            if callable(close):
                close()


class VerlAgentAlfWorldRolloutManager(RolloutManager):
    """Run ALFWorld with prompts, memory and projection from verl-agent."""

    FAILURE_MARKERS = (
        "nothing happens",
        "not a valid command",
        "that's not a verb i recognise",
        "you can't",
        "you cannot",
    )

    def __init__(
        self,
        strategy,
        rollout_group,
        is_same_tokenizer: bool = True,
        generate_kwargs: Optional[Dict[str, Any]] = None,
        env=None,
    ):
        super().__init__(strategy, rollout_group, is_same_tokenizer, generate_kwargs)
        self.batch_size = self.args.rollout.rollout_batch_size
        self.max_turns = self.args.rollout.alfworld_max_turns
        self.history_length = self.args.rollout.alfworld_history_length
        self.prompt_max_len = self.args.data.prompt_max_len
        self.training_mode = getattr(self.args, "scenario", None) == "on_policy_kd"
        if self.training_mode:
            self.script_seed = self.args.train.seed
            self.environment_seed = self.args.train.seed
        else:
            seeds = self.parse_seeds(self.args.rollout.alfworld_eval_seeds)
            self.script_seed = seeds[0]
            self.environment_seed = (
                self.script_seed + self.args.rollout.alfworld_eval_seed_offset
            )
        self.episode_offset = 0
        self._train_episode_offset = 0
        self.env = env if env is not None else self._build_env()

    @staticmethod
    def parse_seeds(value: str) -> List[int]:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
        if not seeds:
            raise ValueError("At least one ALFWorld evaluation seed is required.")
        return seeds

    def _build_env(self):
        try:
            import yaml
            from alfworld.info import ALFWORLD_DATA
        except ImportError as error:
            raise ImportError(
                "The verl-agent-compatible evaluator requires ALFWorld. Install "
                "it with `pip install -e '.[alfworld]'`."
            ) from error

        config_path = os.path.abspath(self.args.rollout.alfworld_config)
        with open(config_path, "r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        os.environ.setdefault("ALFWORLD_DATA", str(ALFWORLD_DATA))
        config.setdefault("general", {})["training_method"] = "dagger"
        config.setdefault("dagger", {}).setdefault("training", {})[
            "max_nb_steps_per_episode"
        ] = self.max_turns
        random.seed(self.environment_seed)
        split = "train" if self.training_mode else self.args.rollout.alfworld_split
        logger.info(
            "Initializing verl-agent ALFWorld protocol: mode=%s split=%s batch=%d "
            "history=%d max_steps=%d environment_seed=%d",
            "train" if self.training_mode else "eval",
            split,
            self.batch_size,
            self.history_length,
            self.max_turns,
            self.environment_seed,
        )
        return _IndependentAlfWorldBatch(
            config=config,
            split=split,
            batch_size=self.batch_size,
            base_seed=self.environment_seed,
        )

    def set_evaluation_batch(self, script_seed: int, episode_offset: int) -> None:
        self.script_seed = script_seed
        self.environment_seed = (
            script_seed + self.args.rollout.alfworld_eval_seed_offset
        )
        self.episode_offset = episode_offset
        setter = getattr(self.env, "set_episode_range", None)
        if callable(setter):
            setter(self.environment_seed, episode_offset)

    @staticmethod
    def _info_values(info: Dict[str, Any], key: str, size: int, default):
        values = info.get(key) if isinstance(info, dict) else None
        if values is None:
            return [default for _ in range(size)]
        if isinstance(values, (str, bytes)):
            return [values for _ in range(size)]
        return list(values)

    @staticmethod
    def _extract_task(observation: str) -> str:
        marker = "Your task is to: "
        start = observation.find(marker)
        if start == -1:
            raise ValueError("Task description not found in ALFWorld observation.")
        return observation[start + len(marker) :].strip()

    def _history_context(self, history: List[Dict[str, str]]) -> str:
        recent = history[-self.history_length :]
        start_index = len(history) - len(recent)
        lines = []
        for local_index, record in enumerate(recent):
            step_number = start_index + local_index + 1
            lines.append(
                f"[Observation {step_number}: '{record['observation']}', "
                f"Action {step_number}: '{record['action']}']"
            )
        return "\n".join(lines)

    def _build_observation_prompt(
        self,
        observation: str,
        admissible_actions: Sequence[str],
        task: str,
        history: List[Dict[str, str]],
        initial: bool,
    ) -> str:
        actions = "\n ".join(
            f"'{action}'" for action in admissible_actions if action != "help"
        )
        if initial or self.history_length <= 0:
            return ALFWORLD_TEMPLATE_NO_HISTORY.format(
                current_observation=observation,
                admissible_actions=actions,
            )
        return ALFWORLD_TEMPLATE_WITH_HISTORY.format(
            task_description=task,
            step_count=len(history),
            history_length=min(len(history), self.history_length),
            action_history=self._history_context(history),
            current_step=len(history) + 1,
            current_observation=observation,
            admissible_actions=actions,
        )

    def _render_prompt(self, content: str, processor=None) -> Tuple[str, bool]:
        if processor is None:
            processor = self.data_processor.student_processor
        template_owner = getattr(processor, "tokenizer", processor)
        kwargs = {}
        if "enable_thinking" in str(getattr(template_owner, "chat_template", "")):
            kwargs["enable_thinking"] = self.args.model.enable_thinking
        prompt = template_owner.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            **kwargs,
        )
        token_ids = template_owner.encode(prompt, add_special_tokens=False)
        if self.prompt_max_len > 0 and len(token_ids) > self.prompt_max_len:
            prompt = template_owner.decode(
                token_ids[-self.prompt_max_len :], skip_special_tokens=False
            )
            return prompt, True
        return prompt, False

    def collect_episodes(
        self,
        prompt_batch: List[Dict[str, Any]],
        global_step: int,
        mode: str = "eval",
        **kwargs,
    ) -> Tuple[List[dict], Dict[str, float]]:
        if mode not in ("train", "eval"):
            raise ValueError(f"Unsupported ALFWorld rollout mode: {mode!r}.")
        if len(prompt_batch) != self.batch_size:
            raise ValueError(
                f"Expected a fixed batch of {self.batch_size}, got {len(prompt_batch)}."
            )

        if mode == "train":
            self.script_seed = self.args.train.seed
            self.environment_seed = self.args.train.seed
            self.episode_offset = self._train_episode_offset
            setter = getattr(self.env, "set_episode_range", None)
            if callable(setter):
                setter(self.environment_seed, self.episode_offset)

        if self.args.train.enable_sleep:
            self.rollout_group.wakeup()
        try:
            observations, reset_info = self.env.reset()
            observations = [str(value) for value in observations]
            admissible = self._info_values(
                reset_info, "admissible_commands", self.batch_size, []
            )
            saved_gamefiles = [
                str(value or "")
                for value in self._info_values(
                    reset_info, "extra.gamefile", self.batch_size, ""
                )
            ]
            episode_gamefiles = list(saved_gamefiles)
            tasks = [self._extract_task(value) for value in observations]
            histories: List[List[Dict[str, str]]] = [
                [] for _ in range(self.batch_size)
            ]
            active = [True] * self.batch_size
            turn_counts = [0] * self.batch_size
            final_won = [False] * self.batch_size
            format_invalid = 0
            inadmissible_count = 0
            environment_failures = 0
            prompt_truncated_count = 0
            records = []
            timing_sums: Dict[str, float] = {}
            sampling_params = kwargs or self.generate_kwargs

            for turn_id in range(self.max_turns):
                request_indices = []
                student_prompts = []
                teacher_prompts = []
                truncated_flags = []
                teacher_processor = self.data_processor._get_teacher_processor()
                for index in range(self.batch_size):
                    if not active[index]:
                        continue
                    content = self._build_observation_prompt(
                        observation=observations[index],
                        admissible_actions=admissible[index],
                        task=tasks[index],
                        history=histories[index],
                        initial=turn_id == 0,
                    )
                    student_prompt, student_truncated = self._render_prompt(content)
                    teacher_prompt, teacher_truncated = self._render_prompt(
                        content, processor=teacher_processor
                    )
                    request_indices.append(index)
                    student_prompts.append(student_prompt)
                    teacher_prompts.append(teacher_prompt)
                    truncated_flags.append(
                        bool(student_truncated or teacher_truncated)
                    )

                if not student_prompts:
                    break
                outputs, timing = self._generate(student_prompts, sampling_params)
                for key, value in timing.items():
                    timing_sums[key] = timing_sums.get(key, 0.0) + float(value)

                actions = ["look"] * self.batch_size
                pending = {}
                for local_index, episode_index in enumerate(request_indices):
                    output = outputs[local_index]
                    action, format_valid = project_alfworld_action(
                        output.get("text", "")
                    )
                    in_admissible = action in admissible[episode_index]
                    actions[episode_index] = action
                    format_invalid += int(not format_valid)
                    inadmissible_count += int(not in_admissible)
                    prompt_truncated_count += int(truncated_flags[local_index])
                    pending[episode_index] = {
                        "stu_prompt": student_prompts[local_index],
                        "tea_prompt": teacher_prompts[local_index],
                        "output": output,
                        "action": action,
                        "format_valid": format_valid,
                        "action_in_admissible": in_admissible,
                        "prompt_truncated": truncated_flags[local_index],
                    }

                next_observations, _, dones, step_info = self.env.step(actions)
                next_observations = [str(value) for value in next_observations]
                next_admissible = self._info_values(
                    step_info, "admissible_commands", self.batch_size, []
                )
                won = self._info_values(step_info, "won", self.batch_size, False)
                step_gamefiles = self._info_values(
                    step_info, "extra.gamefile", self.batch_size, ""
                )

                for episode_index, item in pending.items():
                    feedback = next_observations[episode_index]
                    gamefile = str(
                        step_gamefiles[episode_index]
                        or saved_gamefiles[episode_index]
                    )
                    episode_gamefiles[episode_index] = gamefile
                    env_failed = any(
                        marker in feedback.casefold()
                        for marker in self.FAILURE_MARKERS
                    )
                    environment_failures += int(env_failed)
                    turn_counts[episode_index] += 1
                    final_won[episode_index] = bool(won[episode_index])
                    histories[episode_index].append(
                        {
                            "observation": observations[episode_index],
                            "action": item["action"],
                        }
                    )
                    records.append(
                        {
                            "stu_prompt": item["stu_prompt"],
                            "tea_prompt": item["tea_prompt"],
                            "output": item["output"],
                            "metadata": {
                                "episode_id": (
                                    f"seed{self.script_seed}:"
                                    f"{self.episode_offset + episode_index}"
                                ),
                                "script_seed": self.script_seed,
                                "environment_seed": (
                                    self.environment_seed
                                    + self.episode_offset
                                    + episode_index
                                ),
                                "turn_id": turn_id,
                                "alfworld_action": item["action"],
                                "format_valid": bool(item["format_valid"]),
                                "action_in_admissible": bool(
                                    item["action_in_admissible"]
                                ),
                                "environment_failed": bool(env_failed),
                                "prompt_truncated": bool(
                                    item["prompt_truncated"]
                                ),
                                "episode_done": bool(dones[episode_index]),
                                "episode_won": bool(won[episode_index]),
                                "gamefile": gamefile,
                                "task_type": task_type_from_gamefile(gamefile),
                            },
                        }
                    )
                    if dones[episode_index]:
                        active[episode_index] = False

                observations = next_observations
                admissible = next_admissible
                if not any(active):
                    break

            total_turns = sum(turn_counts)
            prefix = "rollout/alfworld" if mode == "train" else "eval/alfworld"
            metrics = dict(timing_sums)
            metrics.update(
                {
                    f"{prefix}/episodes": float(self.batch_size),
                    f"{prefix}/episodes_succeeded": float(sum(final_won)),
                    f"{prefix}/turns/total": float(total_turns),
                    f"{prefix}/turns/max": float(max(turn_counts, default=0)),
                    f"{prefix}/format_invalid_count": float(format_invalid),
                    f"{prefix}/inadmissible_action_count": float(
                        inadmissible_count
                    ),
                    f"{prefix}/environment_failure_count": float(
                        environment_failures
                    ),
                    f"{prefix}/prompt_truncated_count": float(
                        prompt_truncated_count
                    ),
                }
            )
            for task_type in TASK_TYPES + ("unknown",):
                indices = [
                    index
                    for index, gamefile in enumerate(episode_gamefiles)
                    if task_type_from_gamefile(gamefile) == task_type
                ]
                metrics[f"{prefix}/task/{task_type}/episodes"] = float(
                    len(indices)
                )
                metrics[f"{prefix}/task/{task_type}/succeeded"] = float(
                    sum(final_won[index] for index in indices)
                )
            if mode == "train":
                self._train_episode_offset += self.batch_size
            return records, metrics
        finally:
            if self.args.train.enable_sleep:
                self.rollout_group.sleep()

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if callable(close):
            close()

    def rollout(
        self,
        prompt_batch: List[Dict[str, Any]],
        global_step: int,
        mode: str = "train",
        **kwargs,
    ) -> Tuple[List[dict], Dict[str, float]]:
        """Collect aligned turns and convert them into ordinary KD batches."""
        records, environment_metrics = self.collect_episodes(
            prompt_batch=prompt_batch,
            global_step=global_step,
            mode=mode,
            **kwargs,
        )
        if not records:
            return [], environment_metrics

        sampling_params = kwargs or self.generate_kwargs
        micro_batches, rollout_metrics = self.data_processor.process(
            stu_prompts=[record["stu_prompt"] for record in records],
            tea_prompts=[record["tea_prompt"] for record in records],
            outputs=[record["output"] for record in records],
            labels=[record["metadata"]["episode_won"] for record in records],
            sampling_params=sampling_params,
            global_step=global_step,
            mode=mode,
            sample_metadata=[record["metadata"] for record in records],
        )
        rollout_metrics.update(environment_metrics)
        return micro_batches, rollout_metrics
