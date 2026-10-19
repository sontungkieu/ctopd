"""Single-rank actor integration for complete-window TRUST-B/P calibration."""
from collections import defaultdict

import torch

from .mp_opd_trust import restore_rng


def fit_trust_batch(actor, train_data):
    actor.student.train()
    if actor.args.train.use_dynamic_bsz:
        actor.strategy.accumulated_gradient = len(train_data)
        actor.strategy.step = 0
    algo = actor.kd_algorithm
    prepared = algo.prepare_optimizer_batch(train_data,batch_loader=actor._prepare_micro_batch)
    status = defaultdict(list)
    try:
        for index,batch in enumerate(train_data):
            restore_rng(prepared['rng_states'][index])
            micro = actor._prepare_micro_batch(batch)
            info = algo.training_step(micro)
            for key,value in info.items():
                status[key].append(value.item())
            actor.strategy.backward(info['train/loss'],actor.student,actor.optim)
            status['train/grad_norm'].append(torch.nn.utils.clip_grad_norm_(actor.student.parameters(),float('inf')).item())
            actor.strategy.optimizer_step(actor.optim,actor.student,actor.scheduler)
            if actor.args.kd.use_ema_teacher and actor.strategy.step == 0:
                actor.ema_update()
            for name in ('response_length','total_length'):
                if name in micro:
                    status['rollout/'+name].append(micro[name].mean().item())
            del micro
        restore_rng(prepared['after_rng'])
    finally:
        algo.finish_optimizer_batch()
    status = {k:sum(v)/len(v) for k,v in status.items()}
    status['train/lr'] = actor.scheduler.get_last_lr()[0]
    return {k:actor.strategy.all_reduce(v,op='mean') for k,v in status.items()}


@torch.no_grad()
def evaluate_trust_batch(actor, eval_data):
    """One evaluation-set calibration; no training-state reuse or optimizer step."""
    algo = actor.kd_algorithm
    was_training = actor.student.training
    actor.student.eval()
    batches = list(eval_data)

    def load(batch):
        micro = actor._prepare_micro_batch(batch)
        weight = micro.pop('_eval_weight',1.)
        if float(weight) != 1.:
            raise ValueError('Single-rank TRUST evaluation requires unit sample weights')
        micro['avg_micro_batch_token_num'] = 1.
        return micro

    sums,total = defaultdict(float),0.
    try:
        if not batches:
            return {'eval/num_tokens':0.}
        prepared = algo.prepare_optimizer_batch(batches,batch_loader=load,check_window=False)
        for index,batch in enumerate(batches):
            restore_rng(prepared['rng_states'][index])
            micro = load(batch)
            count = float(micro['stu_loss_mask'].sum())
            metrics = algo.training_step(micro)
            for key,value in metrics.items():
                # Loss already sums tokens; diagnostic values are response averages.
                sums[key] += float(value) * (1. if key in {'train/loss','train/kd_loss'} else count)
            total += count
        restore_rng(prepared['after_rng'])
    finally:
        algo.finish_optimizer_batch()
        actor.student.train(was_training)
    from kdflow.utils.logging_utils import normalize_eval_metrics
    result = {k:actor.strategy.all_reduce(v,op='sum')/max(total,1.) for k,v in sums.items()}
    result['eval/num_tokens'] = total
    return normalize_eval_metrics(result)
