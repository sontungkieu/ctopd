"""Opt-in SimCT MP-OPD adapter; existing CTOPD algorithms are not subclasses.

Numerical kernels live in _mp_opd_port and load only when a new mode is selected.
The contract is sampled path-credit times student NLL, not shared-vocab KL.
"""
from __future__ import annotations

import hashlib
import json
import random

import torch
from torch.utils.checkpoint import checkpoint

from . import register_algorithm


def fixed_partition(n, length):
    if n < 0 or length < 1:
        raise ValueError('n must be nonnegative and length positive')
    return tuple((start, min(start + length, n)) for start in range(0, n, length))


def random_partition(n, minimum, maximum, seed):
    """Exact-cover donor tiling; only the final tail may be shorter than minimum."""
    if n < 0 or not 1 <= minimum <= maximum:
        raise ValueError('Require n >= 0 and 1 <= minimum <= maximum')
    rng, result, cursor = random.Random(int(seed)), [], 0
    while cursor < n:
        high = min(maximum, n - cursor)
        length = rng.randint(min(minimum, high), high)
        result.append((cursor, cursor + length))
        cursor += length
    return tuple(result)


def response_seed(seed, student_ids, teacher_ids):
    # No row/rank/microbatch index: regrouping the same samples cannot retile them.
    payload = json.dumps([int(seed), student_ids, teacher_ids], separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big')


def final_softcap(config):
    config = getattr(config, 'text_config', None) or config
    cap = getattr(config, 'final_logit_softcapping', None)
    if cap is not None and float(cap) <= 0:
        raise ValueError('final_logit_softcapping must be positive')
    return None if cap is None else float(cap)


class MPOnPolicyKD:
    mode = None

    def __init__(self, strategy, student_model, teacher_lm_head,
                 student_tokenizer, teacher_tokenizer, **_kwargs):
        # Lazy imports keep all donor helpers off the existing Strict import path.
        from ._mp_opd_port._mp_opd_atoms import SimCTAtomizer
        from ._mp_opd_port._mp_opd_grass_span import GrassNoiseEstimator, _env_flag

        self.strategy, self.args = strategy, strategy.args
        self.student, self.teacher_head = student_model, teacher_lm_head
        self.kd = self.args.kd
        if self.kd.kd_ratio != 1.0 or self.kd.kd_temperature != 1.0 or self.kd.kd_loss_fn != 'rkl':
            raise ValueError('MP-OPD requires kd_ratio=1, kd_temperature=1, kd_loss_fn=rkl')
        if getattr(self.args, 'scenario', 'on_policy_kd') != 'on_policy_kd':
            raise ValueError('The MP-OPD port supports on-policy KD only')
        world = getattr(self.args.train, 'num_nodes', 1) * getattr(self.args.train, 'num_gpus_per_node', 1)
        if world != 1 or getattr(self.args.model, 'ring_attn_size', 1) != 1:
            raise ValueError('The initial MP-OPD port requires one student rank and ring_attn_size=1')
        self.atomizer = SimCTAtomizer(student_tokenizer, teacher_tokenizer)
        self.student_cap = final_softcap(self.student.model.config)
        self.teacher_cap = getattr(teacher_lm_head, '_mp_opd_softcap', None)
        if self.mode in {'grass_chunk', 'align'} and not _env_flag('MP_OPD_GRASS_EXACT_FP32', True):
            raise ValueError('The port requires MP_OPD_GRASS_EXACT_FP32 enabled for head geometry')
        self.noise = GrassNoiseEstimator(
            rho=self.kd.mp_opd_grass_sigma_rho,
            min_adjacent_pairs=self.kd.mp_opd_grass_sigma_min_pairs,
        ) if self.mode == 'grass_chunk' else None
        self.aligner = None
        if self.mode in {'grass_chunk', 'align'} and self.kd.mp_opd_chunk_source == 'xtoken':
            from ._mp_opd_port.xtoken_upstream_token_aligner import TokenAligner
            self.aligner = TokenAligner(student_tokenizer, teacher_tokenizer, '',
                                       max_comb_len=self.kd.mp_opd_alignment_max_comb_len)

    def state_dict(self):
        return {'version': 1, 'mode': self.mode,
                'contract': {name: value for name, value in vars(self.kd).items() if name.startswith('mp_opd_')},
                'noise': self.noise.state_dict() if self.noise is not None else None}

    def load_state_dict(self, state):
        if (state.get('version') != 1 or state.get('mode') != self.mode
                or state.get('contract') != self.state_dict()['contract']):
            raise ValueError('Incompatible MP-OPD state')
        if self.noise is not None:
            self.noise.load_state_dict(state['noise'])

    @staticmethod
    def _head_logits(head, hidden, softcap, student):
        logits = head(hidden, skip=False) if student else head(hidden)
        return logits if softcap is None else torch.tanh(logits / softcap) * softcap

    def _selected_logps(self, head, hidden, labels, softcap, *, student):
        """Checkpoint token tiles so backward never retains response-wide logits."""
        width = self.args.train.chunked_loss_size or 2048
        if width < 1:
            raise ValueError('chunked_loss_size must be positive')
        pieces = []

        def compute(h, y):
            logits = self._head_logits(head, h, softcap, student)
            return torch.log_softmax(logits.float(), -1).gather(1, y[:, None]).squeeze(1)

        for start in range(0, labels.numel(), width):
            h, y = hidden[start:start + width], labels[start:start + width]
            if student and torch.is_grad_enabled():
                pieces.append(checkpoint(compute, h, y, use_reentrant=False))
            else:
                pieces.append(compute(h, y))
        return torch.cat(pieces) if pieces else hidden.sum(-1)[:0]

    def _partition(self, atoms, student_ids, teacher_ids):
        from ._mp_opd_port._mp_opd_grass_chunk import (
            atom_chunk_ids_from_tokens, chunk_partition, run_chunk_assignment,
        )
        if self.mode == 'atomic':
            return fixed_partition(len(atoms), 1), None, 0
        if self.mode == 'fixed':
            return fixed_partition(len(atoms), self.kd.mp_opd_fixed_span_length), None, 0
        if self.mode == 'random':
            return random_partition(len(atoms), self.kd.mp_opd_random_min_length,
                                    self.kd.mp_opd_random_max_length,
                                    response_seed(self.kd.mp_opd_random_seed, student_ids, teacher_ids)), None, 0
        if self.kd.mp_opd_chunk_source == 'run':
            assignment = run_chunk_assignment(len(atoms), self.kd.mp_opd_chunk_run_length)
        else:
            aligned = self.aligner.align(torch.tensor([student_ids]), torch.tensor([teacher_ids]))
            assignment = atom_chunk_ids_from_tokens(
                aligned.student_chunk_id[0].tolist(),
                [(atom.student_start, atom.student_end) for atom in atoms],
                policy=self.kd.mp_opd_chunk_straddle)
        partition, noncontiguous = chunk_partition(assignment)
        return partition, assignment, noncontiguous

    def _effective_rates(self, credits, atoms, partition, hidden, labels, *, assignment=None, noncontiguous=0):
        from ._mp_opd_port._mp_opd_grass_span import hard_pooled_credits
        if self.mode == 'atomic':
            return credits.rate, {}
        if self.mode in {'fixed', 'random'}:
            return hard_pooled_credits(credits.base_credit, credits.weight, partition).to(credits.rate), {}
        from ._mp_opd_port._mp_opd_grass_chunk import chunk_head_gram
        with torch.no_grad():
            logits = self._head_logits(self.student.model.lm_head, hidden,
                                       self.student_cap, True)
            head = chunk_head_gram(
                logits, hidden.detach(), labels,
                [(atom.student_start, atom.student_end) for atom in atoms], partition,
                selected_log_prob=-credits.student_token_nll,
                softcap=self.student_cap,
                head_bias=getattr(self.student.model.lm_head, 'bias', None) is not None,
                diagonal_only=self.kd.mp_opd_grass_geometry == 'diag')
            if self.mode == 'align':
                from ._mp_opd_port._mp_opd_align import align_chunks, align_metrics
                stats = align_chunks(credits.rate, head.gram, partition, eps_h=self.kd.mp_opd_align_eps_h)
                effective = stats['rate'].to(credits.rate.dtype)
                metrics = align_metrics(credits.rate, effective, head.gram, stats)
            else:
                from ._mp_opd_port._mp_opd_grass_chunk import (
                    apply_chunk_shrinkage, grass_chunk_metrics, grass_chunk_tables,
                )
                from ._mp_opd_port._mp_opd_grass_span import grass_span_ids, grass_credit_residual
                ids = grass_span_ids(partition, len(atoms), credits.rate.device)
                self.noise.update(credits.rate, credits.weight, same_group=ids[:-1] == ids[1:])
                tables = grass_chunk_tables(credits.rate, credits.weight, head.gram,
                                           partition, self.noise.sigma2)
                effective, _ = apply_chunk_shrinkage(credits.rate, credits.weight, tables)
                effective = effective.to(credits.rate.dtype)
                residual = grass_credit_residual(credits.rate, credits.weight, effective)
                if float(residual) > 1e-6 * max(float(credits.base_credit.abs().sum()), 1.0):
                    raise RuntimeError('GRASS-Chunk violated weighted credit conservation')
                metrics = grass_chunk_metrics(tables, credits.rate, credits.weight,
                                              head.gram, effective, residual, assignment=assignment,
                                              noncontiguous_splits=noncontiguous,
                                              source=self.kd.mp_opd_chunk_source)
        return effective, metrics

    def training_step(self, micro_batch):
        from ._mp_opd_port._mp_opd_credit import AtomCreditTensors, soft_partition_loss

        student_mask, teacher_mask = micro_batch['stu_loss_mask'].bool(), micro_batch['tea_loss_mask'].bool()
        if student_mask.shape[0] != teacher_mask.shape[0]:
            raise ValueError('MP-OPD teacher/student response counts differ')
        output = self.student(micro_batch['stu_input_ids'], attention_mask=micro_batch['stu_attn_mask'],
                              allgather_logits=True, ring_attn_group=self.strategy.ring_attn_group,
                              **(micro_batch.get('stu_multi_modal_inputs') or {}))
        hidden = output['hidden_states'][-1][student_mask]
        teacher_hidden = micro_batch['teacher_hiddens']
        if isinstance(teacher_hidden, list):
            teacher_hidden = torch.cat(teacher_hidden, dim=0)
        if teacher_hidden.shape[0] != int(teacher_mask.sum()):
            raise ValueError('Teacher hidden rows do not match the response loss mask')
        teacher_hidden = teacher_hidden.to(self.teacher_head.weight)
        student_labels = micro_batch['stu_input_ids'].roll(-1, dims=1)
        teacher_labels = micro_batch['tea_input_ids'].roll(-1, dims=1)
        total, metrics, responses, atom_count = hidden.sum() * 0.0, {}, 0, 0
        student_offset = teacher_offset = 0
        for row in range(student_mask.shape[0]):
            sy, ty = student_labels[row][student_mask[row]], teacher_labels[row][teacher_mask[row]]
            sh = hidden[student_offset:student_offset + sy.numel()]
            th = teacher_hidden[teacher_offset:teacher_offset + ty.numel()]
            student_offset += sy.numel()
            teacher_offset += ty.numel()
            sid, tid = sy.detach().cpu().tolist(), ty.detach().cpu().tolist()
            atomized = self.atomizer.atomize(sid, tid, sample_id=str(response_seed(0, sid, tid)))
            if not atomized.valid:
                if atomized.failure_reason == 'empty_response':
                    continue
                raise ValueError('MP-OPD atomization failed: ' + atomized.failure_reason)
            atoms = atomized.atoms
            ns, nt = atomized.covered_student_events, atomized.covered_teacher_events
            sy, ty, sh, th = sy[:ns], ty[:nt], sh[:ns], th[:nt]
            slp = self._selected_logps(self.student.model.lm_head, sh, sy, self.student_cap, student=True)
            with torch.no_grad():
                tlp = self._selected_logps(self.teacher_head, th, ty, self.teacher_cap, student=False)
            ss = torch.stack([slp[a.student_start:a.student_end].sum() for a in atoms])
            ts = torch.stack([tlp[a.teacher_start:a.teacher_end].sum() for a in atoms]).detach()
            weight = ss.new_tensor([a.student_token_count for a in atoms])
            base = (ts - ss.detach()).detach()
            credits = AtomCreditTensors(ts, ss.detach(), base, weight, (base / weight).detach(), -ss, -slp)
            partition, assignment, noncontiguous = self._partition(atoms, sid[:ns], tid[:nt])
            effective, extras = self._effective_rates(credits, atoms, partition, sh, sy,
                assignment=assignment, noncontiguous=noncontiguous)
            loss = soft_partition_loss(credits.current_nll, effective)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('MP-OPD produced non-finite loss; no Atomic fallback')
            total = total + loss
            for key, value in extras.items():
                key = 'distill/' + key
                metrics[key] = metrics.get(key, total.detach().new_zeros(())) + value.detach()
            responses += 1
            atom_count += len(atoms)
        denominator = micro_batch['avg_micro_batch_token_num']
        if float(denominator) <= 0:
            raise ValueError('MP-OPD requires a positive shared token normalizer')
        loss = total / denominator
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('MP-OPD produced non-finite normalized loss')
        info = {'train/loss': loss, 'train/kd_loss': loss,
                'distill/mp_opd_valid_responses': loss.detach().new_tensor(float(responses)),
                'distill/mp_opd_atoms': loss.detach().new_tensor(float(atom_count))}
        info.update({key: value / max(responses, 1) for key, value in metrics.items()})
        return info


@register_algorithm('mp_opd_atomic')
class AtomicMPOnPolicyKD(MPOnPolicyKD):
    mode = 'atomic'


@register_algorithm('mp_opd_fixed')
class FixedMPOnPolicyKD(MPOnPolicyKD):
    mode = 'fixed'


@register_algorithm('mp_opd_random')
class RandomMPOnPolicyKD(MPOnPolicyKD):
    mode = 'random'


@register_algorithm('mp_opd_grass_chunk')
class GrassChunkMPOnPolicyKD(MPOnPolicyKD):
    mode = 'grass_chunk'


@register_algorithm('mp_opd_align')
class AlignMPOnPolicyKD(MPOnPolicyKD):
    mode = 'align'
