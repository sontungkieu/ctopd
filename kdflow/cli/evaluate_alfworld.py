"""Evaluate a model checkpoint in ALFWorld without starting training actors."""

import json
import logging
import os

from kdflow.evaluation import AlfWorldEvaluator, VerlAgentAlfWorldEvaluator


logger = logging.getLogger(__name__)


class _EvaluationStrategy:
    """Small args/log adapter required by the shared rollout manager."""

    def __init__(self, args):
        self.args = args

    @staticmethod
    def log(message, level="info", **_kwargs):
        getattr(logger, level, logger.info)(message)

    @staticmethod
    def print(*values):
        logger.info(" ".join(str(value) for value in values))


def evaluate(args):
    if args.rollout.alfworld_eval_protocol == "verl_agent":
        preflight_evaluator = VerlAgentAlfWorldEvaluator(args, None)
        if not preflight_evaluator.has_pending_seeds():
            logger.info(
                "All requested ALFWorld seeds are already complete and "
                "compatible; skipping Ray and model initialization."
            )
            summary = preflight_evaluator.evaluate()
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return summary

    import ray

    from kdflow.ray.placement_group import create_placement_group
    from kdflow.ray.rollout.rollout_group import RolloutActorGroup
    from kdflow.trainer.alfworld_rollout_manager import AlfWorldRolloutManager
    from kdflow.trainer.verl_agent_alfworld_rollout_manager import (
        VerlAgentAlfWorldRolloutManager,
    )

    initialized_ray = False
    if not ray.is_initialized():
        ray.init(
            runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": "WARN",
                },
                "working_dir": os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                ),
            }
        )
        initialized_ray = True

    total_gpus = args.train.num_nodes * args.train.num_gpus_per_node
    rollout_group = None
    rollout_manager = None
    try:
        pg, reordered_bundle_indices, reordered_gpu_ids = create_placement_group(
            total_gpus
        )
        rollout_group = RolloutActorGroup(
            model_path=args.model.student_name_or_path,
            num_actors=args.rollout.rollout_num_engines,
            tp_size=args.rollout.rollout_tp_size,
            num_gpus_per_node=args.train.num_gpus_per_node,
            enable_memory_saver=True,
            mem_fraction_static=args.rollout.rollout_mem_fraction_static,
            num_gpus_per_actor=0.01,
            pg=(pg, reordered_bundle_indices, reordered_gpu_ids),
            extra_server_args={"disable_piecewise_cuda_graph": True},
        )
        if args.train.enable_sleep:
            rollout_group.sleep()

        strategy = _EvaluationStrategy(args)
        generate_kwargs = {
            "max_new_tokens": args.rollout.generate_max_len,
            "temperature": args.rollout.temperature,
            "top_p": args.rollout.top_p,
        }
        if args.rollout.alfworld_eval_protocol == "verl_agent":
            manager_class = VerlAgentAlfWorldRolloutManager
            evaluator_class = VerlAgentAlfWorldEvaluator
        else:
            manager_class = AlfWorldRolloutManager
            evaluator_class = AlfWorldEvaluator
        rollout_manager = manager_class(
            strategy=strategy,
            rollout_group=rollout_group,
            is_same_tokenizer=True,
            generate_kwargs=generate_kwargs,
        )
        summary = evaluator_class(args, rollout_manager).evaluate()
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary
    finally:
        if rollout_manager is not None:
            rollout_manager.close()
        if rollout_group is not None:
            rollout_group.shutdown()
        if initialized_ray:
            ray.shutdown()


if __name__ == "__main__":
    from kdflow.arguments import init_args

    evaluate(init_args("alfworld_eval"))
