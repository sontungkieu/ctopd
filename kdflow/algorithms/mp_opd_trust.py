"""Isolated TRUST-R/B/P adapters; B/P require a complete optimizer window."""
import math
import random

import numpy as np
import torch

from . import register_algorithm
from .mp_opd import MPOnPolicyKD, fixed_partition


def capture_rng():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)


def restore_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    if state[3] is not None:
        torch.cuda.set_rng_state_all(state[3])


class TrustMPOnPolicyKD(MPOnPolicyKD):
    mode = 'atomic'
    trust_mode = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from ._mp_opd_port._mp_opd_grass_span import _env_flag
        self.mode = self.trust_mode
        self.requires_optimizer_batch = self.mode in {'trust_b', 'trust_p'}
        self.eps_g = float(self.kd.mp_opd_trust_eps_g)
        if not math.isfinite(self.eps_g) or self.eps_g < 0:
            raise ValueError('TRUST eps_g must be finite and nonnegative')
        if self.kd.mp_opd_grass_geometry != 'exact_head' or not _env_flag('MP_OPD_GRASS_EXACT_FP32', True):
            raise ValueError('TRUST requires exact_head with exact FP32 geometry enabled')
        if self.kd.mp_opd_trust_vocab_chunk < 1:
            raise ValueError('TRUST vocabulary tile must be positive')
        self._collector = self._result = None
        self._normalizer = None
        self._batch_metrics = {}

    def _partition(self, atoms, student_ids, teacher_ids):
        return fixed_partition(len(atoms), 1), None, 0

    def training_step(self, batch):
        if self.requires_optimizer_batch:
            if self._collector is None and self._result is None:
                raise RuntimeError('TRUST-B/P requires prepare_optimizer_batch before scoring/backward')
            normalizer = float(batch['avg_micro_batch_token_num'])
            if not math.isfinite(normalizer) or normalizer <= 0:
                raise ValueError('TRUST-B/P requires a finite positive common normalizer')
            if self._normalizer is not None and normalizer != self._normalizer:
                raise ValueError('TRUST-B/P requires one common normalizer across the window')
            self._normalizer = normalizer
        return super().training_step(batch)

    def _effective_rates(self, credits, atoms, partition, hidden, labels, **_kwargs):
        from ._mp_opd_port._mp_opd_trust import (strict_mask, trust_calibrate,
            trust_response_metrics, gram_affordable)
        from ._mp_opd_port._mp_opd_grass_chunk import chunk_head_gram
        strict = strict_mask([a.boundary_type for a in atoms]).to(credits.rate.device)
        ranges = tuple((a.student_start, a.student_end) for a in atoms)
        if self._collector is not None:
            with torch.no_grad():
                logits = self._head_logits(self.student.model.lm_head, hidden, self.student_cap, True)
                self._collector.add_response(logits, hidden.detach(), labels, ranges,
                    credits.rate, strict, -credits.student_token_nll)
            return torch.zeros_like(credits.rate), {}
        if self.requires_optimizer_batch:
            result, metrics = self._result, self._batch_metrics
        else:
            with torch.no_grad():
                logits = self._head_logits(self.student.model.lm_head, hidden, self.student_cap, True)
                if not gram_affordable(labels.numel(), logits.shape[-1], logits.device):
                    raise RuntimeError('TRUST-R exact Gram exceeds memory guard; no silent Atomic fallback')
                head = chunk_head_gram(logits, hidden.detach(), labels, ranges, ((0,len(atoms)),),
                    selected_log_prob=-credits.student_token_nll, softcap=self.student_cap,
                    head_bias=getattr(self.student.model.lm_head,'bias',None) is not None)
                q, h = credits.rate.detach().float(), head.gram.detach().float()
                result = trust_calibrate(q[strict], q[~strict], h[strict][:,strict],
                    h[~strict][:,~strict], h[strict][:,~strict], eps_g=self.eps_g)
                raw = trust_response_metrics(result,strict,credits.weight,scope='response')
                metrics = {'mp_opd_trust_r_'+k: credits.rate.new_tensor(v) for k,v in raw.items()}
        if self.mode == 'trust_p':
            coefficients = torch.where(strict, float(result.strict_scale), 1.).to(credits.rate)
        else:
            coefficients = torch.where(strict, 1., float(result.lam)).to(credits.rate)
        return credits.rate.detach()*coefficients, metrics

    def prepare_optimizer_batch(self, batches, *, batch_loader=lambda x:x, check_window=True):
        from ._mp_opd_port._mp_opd_trust_batch import TrustHeadAccumulator
        from ._mp_opd_port._mp_opd_trust import trust_batch_metrics
        from ._mp_opd_port._mp_opd_trust_p import trust_p_calibrate_scalars, trust_p_batch_metrics
        if not self.requires_optimizer_batch:
            return None
        if self._collector is not None or self._result is not None:
            raise RuntimeError('TRUST optimizer window is already active')
        if not batches:
            raise ValueError('TRUST requires a nonempty optimizer window')
        if check_window and (len(batches) != self.strategy.accumulated_gradient or self.strategy.step != 0):
            raise ValueError('TRUST-B/P requires exactly one complete optimizer accumulation window at step0')
        before = capture_rng()
        states = []
        self._normalizer = None
        collector = TrustHeadAccumulator(softcap=self.student_cap,
            head_bias=getattr(self.student.model.lm_head,'bias',None) is not None,
            vocab_chunk=self.kd.mp_opd_trust_vocab_chunk, force_fp32=self.mode == 'trust_p')
        self._collector = collector
        try:
            with torch.no_grad():
                for batch in batches:
                    states.append(capture_rng())
                    self.training_step(batch_loader(batch))
            after = capture_rng()
            options = {'calibrate':trust_p_calibrate_scalars} if self.mode == 'trust_p' else {}
            self._result = collector.finalize(eps_g=self.eps_g, distributed=False, **options)
            records = collector.responses
            ns = sum(r['strict_atoms'] for r in records)
            nm = sum(r['mismatch_atoms'] for r in records)
            counts = dict(strict_span_count=ns,mismatch_span_count=nm,strict_atom_count=ns,mismatch_atom_count=nm,
                strict_token_count=sum(r['strict_tokens'] for r in records),
                mismatch_token_count=sum(r['mismatch_tokens'] for r in records),
                strict_span_fraction=ns/max(ns+nm,1),mismatch_span_fraction=nm/max(ns+nm,1))
            metric_fn = trust_p_batch_metrics if self.mode == 'trust_p' else trust_batch_metrics
            self._batch_metrics = {f'mp_opd_{self.mode}_'+k: torch.tensor(float(v),device=self.teacher_head.weight.device,dtype=torch.float32)
                                   for k,v in metric_fn(self._result,counts).items()}
            prefix = 'mp_opd_'+self.mode+'_'
            self._batch_metrics.update({prefix+'full_batch':torch.tensor(1.,device=self.teacher_head.weight.device),
                prefix+'calibration_response_count':torch.tensor(float(len(records)),device=self.teacher_head.weight.device)})
            return {'rng_states':states,'after_rng':after}
        except BaseException:
            self.finish_optimizer_batch()
            raise
        finally:
            self._collector = None
            restore_rng(before)

    def finish_optimizer_batch(self):
        self._collector = self._result = None
        self._normalizer = None
        self._batch_metrics = {}


@register_algorithm('mp_opd_trust_r')
class TrustRMPOnPolicyKD(TrustMPOnPolicyKD):
    trust_mode = 'trust_r'


@register_algorithm('mp_opd_trust_b')
class TrustBMPOnPolicyKD(TrustMPOnPolicyKD):
    trust_mode = 'trust_b'


@register_algorithm('mp_opd_trust_p')
class TrustPMPOnPolicyKD(TrustMPOnPolicyKD):
    trust_mode = 'trust_p'
