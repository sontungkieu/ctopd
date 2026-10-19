import os

import ray

from kdflow.ray.train.teacher_group import TeacherActorGroup
from kdflow.ray.train.student_group import StudentActorGroup
from kdflow.ray.rollout.rollout_group import RolloutActorGroup
from kdflow.ray.placement_group import create_placement_group
from kdflow.trainer import OnPolicyKDTrainer
from kdflow.datasets import (
    AlfWorldEpisodeDataset,
    PromptDataset,
)
from kdflow.datasets.utils import blending_datasets
from kdflow.models.utils import check_tokenizer_identical
from kdflow.backend import get_strategy
from kdflow.arguments import init_args
from kdflow.mp_opd_modes import MP_OPD_ALGORITHMS
from kdflow.arguments.training_args import resolve_max_steps
from kdflow.utils.distributed_sampler import DistributedSampler
from kdflow.utils.utils import get_tokenizer, load_custom_eval_fn


def train(args):
    if args.kd.kd_algorithm in MP_OPD_ALGORITHMS and (
        args.train.num_nodes * args.train.num_gpus_per_node != 1
        or args.model.ring_attn_size != 1
    ):
        raise ValueError('The initial MP-OPD port requires one student rank and ring_attn_size=1')
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init(
            runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": "WARN"
                },
                "working_dir": os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            }
        )
    
    strategy = get_strategy(args)
    strategy.print(args)

    student_tokenizer = get_tokenizer(
        args.model.student_name_or_path,
        use_fast=not args.model.disable_fast_tokenizer,
    )
    
    # Create placement group for resource allocation
    num_gpus = args.train.num_nodes * args.train.num_gpus_per_node
    pg, reordered_bundle_indices, reordered_gpu_ids = create_placement_group(num_gpus)
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
    
    teacher_model = TeacherActorGroup(
        strategy,
        num_gpus,
        num_gpus_per_node=args.train.num_gpus_per_node,
        num_gpus_per_actor=0.01,
        pg=(pg, reordered_bundle_indices, reordered_gpu_ids),

    )
    student_model = StudentActorGroup(
        args,
        args.train.num_nodes,
        args.train.num_gpus_per_node,
        pg=(pg, reordered_bundle_indices),
        num_gpus_per_actor=0.01,
    )
    
    # Initialize the teacher tokenizer after inference actors are configured.
    teacher_tokenizer = get_tokenizer(
        args.model.teacher_name_or_path,
        use_fast=not args.model.disable_fast_tokenizer,
    )
    tokenizer_info = check_tokenizer_identical(student_tokenizer, teacher_tokenizer)
    strategy.print(f"Tokenizers {tokenizer_info}")
    if not tokenizer_info.vocab_identical and args.kd.kd_algorithm not in {
        "simple_ctkd",
        "byte_align_ctkd",
        "topk_byte_align_ctkd",
        *MP_OPD_ALGORITHMS,
    }:
        raise ValueError(
            "Cross-tokenizer distillation supports simple_ctkd, "
            "byte_align_ctkd, topk_byte_align_ctkd, or an opt-in mp_opd_* mode."
        )
    
    environment_rollout = bool(args.rollout.alfworld_config)

    # Interactive environments sample tasks on reset. Their lightweight
    # datasets are therefore only epoch clocks; ordinary on-policy KD keeps the
    # existing static prompt dataset path.
    if args.rollout.alfworld_config:
        train_dataset = AlfWorldEpisodeDataset(
            args.rollout.alfworld_episodes_per_epoch
        )
    else:
        train_data = blending_datasets(
            args.data.train_dataset_path,
            args.data.train_dataset_probs,
            strategy,
            args.train.seed,
            max_count=args.data.max_samples,
            dataset_split=args.data.train_split,
        )
        train_data = train_data.select(
            range(min(args.data.max_samples, len(train_data)))
        )
        train_dataset = PromptDataset(
            train_data,
            strategy,
            tokenizer_info=tokenizer_info,
            max_data_num=args.data.max_samples,
            input_template=args.data.input_template,
            num_processors=args.data.preprocess_num_workers,
        )
    
    sampler = DistributedSampler(train_dataset, num_replicas=1, rank=0, shuffle=True, seed=args.train.seed, drop_last=True)
    train_dataloader = strategy.setup_dataloader(
        train_dataset,
        args.rollout.rollout_batch_size,
        True,
        False,
        collate_fn=train_dataset.collate_fn,
        sampler=sampler,
    )
    
    # Load and prepare evaluation dataset (optional)
    eval_dataloader = None
    if getattr(args.data, "eval_dataset_path", None) and not environment_rollout:
        eval_data = blending_datasets(
            args.data.eval_dataset_path,
            None,
            strategy,
            dataset_split=args.data.eval_split,
        )
        eval_data = eval_data.select(range(min(args.data.max_samples, len(eval_data))))
        eval_dataset = PromptDataset(
            eval_data,
            strategy,
            tokenizer_info=tokenizer_info,
            input_template=args.data.input_template,
            num_processors=args.data.preprocess_num_workers,
        )
        eval_dataloader = strategy.setup_dataloader(
            eval_dataset,
            batch_size=args.rollout.rollout_batch_size,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
            collate_fn=eval_dataset.collate_fn,
        )
    
    # Calculate max training steps
    num_rollout_iters_per_epoch = len(train_dataset) // args.rollout.rollout_batch_size
    if environment_rollout:
        max_turn_samples = (
            args.rollout.rollout_batch_size * args.rollout.alfworld_max_turns
        )
        num_update_steps_per_rollout = max(
            1,
            (max_turn_samples + args.train.train_batch_size - 1)
            // args.train.train_batch_size,
        )
    else:
        num_update_steps_per_rollout = max(
            1,
            args.rollout.rollout_batch_size
            * args.rollout.n_samples_per_prompt
            // args.train.train_batch_size,
        )
    max_rollout_iters = resolve_max_steps(
        args.train.max_steps,
        args.train.num_epochs,
        num_rollout_iters_per_epoch,
    )
    strategy.log(f"Max training iterations: {max_rollout_iters}")
    
    # Initialize student model on all workers
    ray.get(student_model.async_init_model_from_pretrained(
        strategy, (max_rollout_iters * num_update_steps_per_rollout), 
        teacher_tokenizer=teacher_tokenizer, 
        tokenizer_info=tokenizer_info,
    ))
    strategy.log("Models initialized on all student actors")
    
    generate_kwargs = {
        "max_new_tokens": args.rollout.generate_max_len,
        "temperature": args.rollout.temperature,
        "top_p": args.rollout.top_p,
    }
    
    trainer = None
    try:
        trainer = OnPolicyKDTrainer(
            strategy=strategy,
            student_model=student_model,
            teacher_model=teacher_model,
            rollout_group=rollout_group,
            is_same_tokenizer=tokenizer_info.is_identical,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            max_rollout_iters=max_rollout_iters,
            num_rollout_iters_per_epoch=num_rollout_iters_per_epoch,
            generate_kwargs=generate_kwargs,
            custom_eval_fn=(
                load_custom_eval_fn(args.data.custom_eval_fn)
                if args.data.custom_eval_fn
                else None
            ),
        )
        trainer.fit()
        ray.get(student_model.async_save_model())
        strategy.log("Training completed and model saved.")
    finally:
        if trainer is not None:
            trainer.close()
        teacher_model.shutdown()
        rollout_group.shutdown()


if __name__ == "__main__":
    args = init_args("on_policy_kd")
    train(args)
