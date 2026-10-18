import os
import random
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from kdflow.trainer.rollout_manager import RolloutManager
from kdflow.utils.logging_utils import init_logger


logger = init_logger(__name__)


ALFWORLD_SYSTEM_PROMPT = """You are an embodied agent solving a household task in the ALFWorld text environment.
You receive an Observation and a list of Admissible commands. On every turn reply with exactly two lines:
Thought: <one short sentence of reasoning>
Action: <one command, copied verbatim from the Admissible list when possible>
Only the Action line is executed. Do not write anything after the Action line."""


class AlfWorldRolloutManager(RolloutManager):
    """Collect batched multi-turn ALFWorld trajectories from the student.

    Every assistant turn is emitted as an independent prompt/response sample.
    Consequently the existing cross-tokenizer processor masks only that turn's
    response, and byte alignment can never cross an environment boundary.
    """

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
        is_same_tokenizer: bool,
        generate_kwargs: Optional[Dict[str, Any]] = None,
        env=None,
    ):
        super().__init__(
            strategy=strategy,
            rollout_group=rollout_group,
            is_same_tokenizer=is_same_tokenizer,
            generate_kwargs=generate_kwargs,
        )
        self.strategy = strategy
        self.max_turns = self.args.rollout.alfworld_max_turns
        self.admissible_limit = self.args.rollout.alfworld_admissible_limit
        self.batch_size = self.args.rollout.rollout_batch_size
        self.env = env if env is not None else self._build_env()

    def _build_env(self):
        config_path = os.path.abspath(self.args.rollout.alfworld_config)
        try:
            import yaml
            from alfworld.agents.environment import get_environment
            from alfworld.info import ALFWORLD_DATA
        except ImportError as error:
            raise ImportError(
                "ALFWorld multi-turn rollout requires the text environment. "
                "Install it with `pip install -e '.[alfworld]'` and run "
                "`alfworld-download` before training."
            ) from error

        with open(config_path, "r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        os.environ.setdefault("ALFWORLD_DATA", str(ALFWORLD_DATA))

        # AlfredTWEnv reads the episode cap from the selected training-method
        # section. DAgger is used only to select that config path; KDFlow never
        # consumes ALFWorld's expert action.
        config.setdefault("general", {})["training_method"] = "dagger"
        seed = self.args.train.seed
        config["general"]["random_seed"] = seed
        random.seed(seed)
        config.setdefault("dagger", {}).setdefault("training", {})[
            "max_nb_steps_per_episode"
        ] = self.max_turns

        env_type = config.get("env", {}).get("type", "AlfredTWEnv")
        if env_type != "AlfredTWEnv":
            raise ValueError(
                "The first multi-turn KD implementation supports the text-only "
                f"AlfredTWEnv, got {env_type!r}."
            )
        split = self.args.rollout.alfworld_split
        logger.info(
            "Initializing ALFWorld: config=%s split=%s batch_size=%d max_turns=%d",
            config_path,
            split,
            self.batch_size,
            self.max_turns,
        )
        env = get_environment(env_type)(config, train_eval=split).init_env(
            batch_size=self.batch_size
        )
        seed_env = getattr(env, "seed", None)
        if callable(seed_env):
            seed_env(seed)
        return env

    @staticmethod
    def parse_action(
        response: str, admissible_commands: Sequence[str]
    ) -> Tuple[str, bool]:
        """Return an executable ALFWorld command and whether parsing failed."""
        matches = re.findall(
            r"(?im)^\s*action\s*:\s*(.*?)\s*$", response or ""
        )
        candidate = matches[-1] if matches else (response or "").strip()
        candidate = re.sub(r"</?action>", "", candidate, flags=re.IGNORECASE)
        candidate = candidate.splitlines()[0].strip() if candidate else ""

        canonical = {
            str(command).strip().casefold(): str(command).strip()
            for command in admissible_commands
        }
        if candidate.casefold() in canonical:
            return canonical[candidate.casefold()], False
        if candidate:
            # TextWorld accepts free-form commands and returns deterministic
            # feedback for invalid ones, which is preferable to changing the
            # student's state trajectory behind its back.
            return candidate, True
        # Empty strings are not accepted uniformly across TextWorld versions.
        # "look" safely advances the batched environment while the sample is
        # still marked as a deterministic parser failure.
        return "look", True

    def _render_prompt(self, messages: List[Dict[str, str]], processor) -> str:
        template_owner = processor
        if not hasattr(template_owner, "apply_chat_template"):
            template_owner = getattr(processor, "tokenizer", processor)
        kwargs = {}
        if "enable_thinking" in str(
            getattr(template_owner, "chat_template", "")
        ):
            kwargs["enable_thinking"] = self.args.model.enable_thinking
        return template_owner.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **kwargs,
        )

    @staticmethod
    def _tokenizer(processor):
        return getattr(processor, "tokenizer", processor)

    def _fits_context(self, student_prompt: str, teacher_prompt: str) -> bool:
        prompt_budget = self.args.data.max_len - int(
            self.generate_kwargs["max_new_tokens"]
        )
        student_len = len(
            self._tokenizer(self.data_processor.student_processor).encode(
                student_prompt, add_special_tokens=False
            )
        )
        teacher_processor = self.data_processor._get_teacher_processor()
        teacher_len = len(
            self._tokenizer(teacher_processor).encode(
                teacher_prompt, add_special_tokens=False
            )
        )
        return student_len <= prompt_budget and teacher_len <= prompt_budget

    def _format_observation(
        self,
        observation: str,
        admissible_commands: Sequence[str],
        initial: bool = False,
    ) -> str:
        commands = [str(command) for command in admissible_commands]
        shown = commands
        suffix = ""
        if self.admissible_limit > 0 and len(commands) > self.admissible_limit:
            shown = commands[: self.admissible_limit]
            suffix = f" (+{len(commands) - len(shown)} more)"
        heading = "Task and initial observation" if initial else "Observation"
        return (
            f"{heading}:\n{observation}\n"
            f"Admissible commands{suffix}: {'; '.join(shown)}"
        )

    @staticmethod
    def _info_list(info: Dict[str, Any], key: str, batch_size: int, default):
        values = info.get(key) if isinstance(info, dict) else None
        if values is None:
            return [default for _ in range(batch_size)]
        return list(values)

    def collect_episodes(
        self,
        prompt_batch: List[Dict[str, Any]],
        global_step: int,
        mode: str = "train",
        **kwargs,
    ) -> tuple[List[dict], Dict[str, float]]:
        """Collect raw ALFWorld turns without running teacher or student updates.

        This is the shared environment loop used by both on-policy training and
        the standalone evaluator. ``prompt_batch`` is an epoch clock only; the
        ALFWorld environment chooses the actual games when it is reset.
        """
        if mode not in ("train", "eval"):
            raise ValueError(f"Unsupported ALFWorld rollout mode: {mode!r}")
        if len(prompt_batch) != self.batch_size:
            raise ValueError(
                "ALFWorld requires a full fixed-size rollout batch: expected "
                f"{self.batch_size}, got {len(prompt_batch)}."
            )

        metric_prefix = "rollout" if mode == "train" else "eval"

        should_sleep = self.args.train.enable_sleep
        if should_sleep:
            self.rollout_group.wakeup()

        try:
            observations, info = self.env.reset()
            observations = list(observations)
            admissible = self._info_list(
                info, "admissible_commands", self.batch_size, []
            )
            student_messages = []
            teacher_messages = []
            for index in range(self.batch_size):
                initial_user = self._format_observation(
                    observations[index], admissible[index], initial=True
                )
                messages = [
                    {"role": "system", "content": ALFWORLD_SYSTEM_PROMPT},
                    {"role": "user", "content": initial_user},
                ]
                student_messages.append(list(messages))
                teacher_messages.append(list(messages))

            active = [True] * self.batch_size
            turn_counts = [0] * self.batch_size
            final_won = [False] * self.batch_size
            context_truncated = [False] * self.batch_size
            invalid_actions = 0
            environment_failures = 0
            records = []
            timing_sums: Dict[str, float] = {}

            sampling_params = kwargs or self.generate_kwargs
            teacher_processor = self.data_processor._get_teacher_processor()

            for turn_id in range(self.max_turns):
                request_indices = []
                stu_prompts = []
                tea_prompts = []
                for index in range(self.batch_size):
                    if not active[index]:
                        continue
                    stu_prompt = self._render_prompt(
                        student_messages[index],
                        self.data_processor.student_processor,
                    )
                    tea_prompt = self._render_prompt(
                        teacher_messages[index], teacher_processor
                    )
                    if not self._fits_context(stu_prompt, tea_prompt):
                        active[index] = False
                        context_truncated[index] = True
                        continue
                    request_indices.append(index)
                    stu_prompts.append(stu_prompt)
                    tea_prompts.append(tea_prompt)

                if not request_indices:
                    break

                outputs, timing = self._generate(stu_prompts, sampling_params)
                for key, value in timing.items():
                    timing_sums[key] = timing_sums.get(key, 0.0) + float(value)

                actions = ["look"] * self.batch_size
                pending = {}
                for local_index, episode_index in enumerate(request_indices):
                    output = outputs[local_index]
                    action, parser_failed = self.parse_action(
                        output.get("text", ""), admissible[episode_index]
                    )
                    actions[episode_index] = action
                    invalid_actions += int(parser_failed)
                    pending[episode_index] = {
                        "stu_prompt": stu_prompts[local_index],
                        "tea_prompt": tea_prompts[local_index],
                        "output": output,
                        "action": action,
                        "parser_failed": parser_failed,
                    }

                next_observations, _, dones, next_info = self.env.step(actions)
                next_observations = list(next_observations)
                dones = list(dones)
                next_admissible = self._info_list(
                    next_info, "admissible_commands", self.batch_size, []
                )
                won = self._info_list(next_info, "won", self.batch_size, False)
                gamefiles = self._info_list(
                    next_info, "extra.gamefile", self.batch_size, ""
                )

                for episode_index, item in pending.items():
                    feedback = str(next_observations[episode_index])
                    env_failed = any(
                        marker in feedback.casefold()
                        for marker in self.FAILURE_MARKERS
                    )
                    environment_failures += int(env_failed)
                    turn_counts[episode_index] += 1
                    final_won[episode_index] = bool(won[episode_index])
                    records.append(
                        {
                            **item,
                            "metadata": {
                                "episode_id": f"{global_step}:{episode_index}",
                                "turn_id": turn_id,
                                "alfworld_action": item["action"],
                                "parser_failed": bool(item["parser_failed"]),
                                "environment_failed": bool(env_failed),
                                "deterministic_failure": bool(
                                    item["parser_failed"] or env_failed
                                ),
                                "episode_done": bool(dones[episode_index]),
                                "episode_won": bool(won[episode_index]),
                                "gamefile": str(gamefiles[episode_index]),
                            },
                        }
                    )

                    if dones[episode_index]:
                        active[episode_index] = False
                        continue
                    assistant_message = {
                        "role": "assistant",
                        "content": item["output"].get("text", ""),
                    }
                    next_user = {
                        "role": "user",
                        "content": self._format_observation(
                            feedback,
                            next_admissible[episode_index],
                            initial=False,
                        ),
                    }
                    student_messages[episode_index].extend(
                        [assistant_message, next_user]
                    )
                    teacher_messages[episode_index].extend(
                        [dict(assistant_message), dict(next_user)]
                    )

                observations = next_observations
                admissible = next_admissible
                if not any(active):
                    break

            if not records:
                return [], {
                    f"{metric_prefix}/alfworld/episodes": float(self.batch_size),
                    f"{metric_prefix}/alfworld/episodes_succeeded": 0.0,
                    f"{metric_prefix}/alfworld/turns/total": 0.0,
                    f"{metric_prefix}/alfworld/context_truncated_count": float(
                        self.batch_size
                    ),
                    f"{metric_prefix}/alfworld/context_truncated_ratio": 1.0,
                }

            total_turns = sum(turn_counts)
            metrics = dict(timing_sums)
            metrics.update(
                {
                    f"{metric_prefix}/alfworld/episodes": float(self.batch_size),
                    f"{metric_prefix}/alfworld/episodes_succeeded": float(
                        sum(final_won)
                    ),
                    f"{metric_prefix}/alfworld/success_rate": (
                        sum(final_won) / self.batch_size
                    ),
                    f"{metric_prefix}/alfworld/turns/total": float(total_turns),
                    f"{metric_prefix}/alfworld/turns/mean": (
                        total_turns / self.batch_size
                    ),
                    f"{metric_prefix}/alfworld/turns/max": float(
                        max(turn_counts)
                    ),
                    f"{metric_prefix}/alfworld/invalid_action_count": float(
                        invalid_actions
                    ),
                    f"{metric_prefix}/alfworld/invalid_action_ratio": (
                        invalid_actions / max(total_turns, 1)
                    ),
                    f"{metric_prefix}/alfworld/environment_failure_count": float(
                        environment_failures
                    ),
                    f"{metric_prefix}/alfworld/environment_failure_ratio": (
                        environment_failures / max(total_turns, 1)
                    ),
                    f"{metric_prefix}/alfworld/context_truncated_count": float(
                        sum(context_truncated)
                    ),
                    f"{metric_prefix}/alfworld/context_truncated_ratio": (
                        sum(context_truncated) / self.batch_size
                    ),
                }
            )
            return records, metrics
        finally:
            if should_sleep:
                self.rollout_group.sleep()

    def rollout(
        self,
        prompt_batch: List[Dict[str, Any]],
        global_step: int,
        mode: str = "train",
        **kwargs,
    ) -> tuple[List[dict], Dict[str, float]]:
        """Collect ALFWorld turns and convert them to KD/evaluation batches."""
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

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if callable(close):
            close()
