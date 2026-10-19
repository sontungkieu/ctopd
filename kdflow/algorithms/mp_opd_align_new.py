"""Isolated donor ALIGN-NEW; no change to the ALIGN v1/GRASS adapter."""
import math

import torch

from . import register_algorithm
from .mp_opd import MPOnPolicyKD


@register_algorithm('mp_opd_align_new')
class AlignNewMPOnPolicyKD(MPOnPolicyKD):
    # Let the unchanged parent initialize the same chunk aligner as ALIGN v1.
    mode = 'align'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = 'align_new'
        self.rho = float(self.kd.mp_opd_align_new_min_descent_fraction)
        self.eps_g = float(self.kd.mp_opd_align_new_eps_g)
        if self.kd.mp_opd_grass_geometry != 'exact_head':
            raise ValueError('ALIGN-NEW requires exact_head geometry')
        if not math.isfinite(self.rho) or not 0 <= self.rho <= 1:
            raise ValueError('ALIGN-NEW minimum descent fraction must be in [0,1]')
        if not math.isfinite(self.eps_g) or self.eps_g < 0:
            raise ValueError('ALIGN-NEW eps_g must be finite and nonnegative')

    def _effective_rates(self, credits, atoms, partition, hidden, labels, **_kwargs):
        from ._mp_opd_port._mp_opd_align import align_chunks, align_metrics
        from ._mp_opd_port._mp_opd_align_new import align_new_guard
        from ._mp_opd_port._mp_opd_grass_chunk import chunk_head_gram
        from ._mp_opd_port._mp_opd_head_pair import head_pair_geometry

        ranges = tuple((a.student_start, a.student_end) for a in atoms)
        with torch.no_grad():
            logits = self._head_logits(self.student.model.lm_head, hidden, self.student_cap, True)
            has_bias = getattr(self.student.model.lm_head, 'bias', None) is not None
            # Reconstruct the same logits as the credit path before disabling
            # ambient autocast for donor geometry/local coefficient arithmetic.
            with torch.autocast(device_type=hidden.device.type, enabled=False):
                head = chunk_head_gram(logits, hidden.detach(), labels, ranges, partition,
                    selected_log_prob=-credits.student_token_nll, softcap=self.student_cap,
                    head_bias=has_bias, diagonal_only=False)
                stats = align_chunks(credits.rate, head.gram, partition, eps_h=self.kd.mp_opd_align_eps_h)
            candidate = stats['rate'].to(credits.rate.dtype)
            metrics = align_metrics(credits.rate, candidate, head.gram, stats)
            pair = head_pair_geometry(logits, hidden.detach(), labels, ranges, credits.rate, candidate,
                selected_log_prob=-credits.student_token_nll, softcap=self.student_cap, head_bias=has_bias)
            effective, global_metrics = align_new_guard(credits.rate, candidate,
                pair.first_norm2, pair.second_norm2, pair.dot,
                min_descent_fraction=self.rho, eps_g=self.eps_g)
            metrics = {k.replace('mp_opd_align_', 'mp_opd_align_new_local_'): v for k, v in metrics.items()}
            metrics.update(global_metrics)
        return effective, metrics
