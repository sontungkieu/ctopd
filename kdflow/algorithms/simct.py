"""Author-code SimCT baseline, integrated with CTOPD's hidden-state interface."""
import torch
from torch.utils.checkpoint import checkpoint

from . import register_algorithm


@register_algorithm('simct')
class SimCT:
    def __init__(self, strategy, student_model, teacher_lm_head,
                 student_tokenizer, teacher_tokenizer, **kwargs):
        # Lazy: selecting Strict/MP never imports the author's implementation.
        from ._simct_upstream.span_ctkd import SpanCrossTokenizerKD
        self.args = strategy.args
        if self.args.scenario != 'on_policy_kd':
            raise ValueError('SimCT baseline supports on-policy KD only')
        if (self.args.kd.kd_ratio != 1 or self.args.kd.kd_temperature != 1
                or self.args.kd.kd_loss_fn != 'rkl'):
            raise ValueError('SimCT baseline requires kd_ratio=1, kd_temperature=1, kd_loss_fn=rkl')
        if (self.args.train.num_nodes * self.args.train.num_gpus_per_node != 1
                or self.args.model.ring_attn_size != 1):
            raise ValueError('SimCT baseline initially requires one student rank and ring_attn_size=1')
        self.student, self.teacher_head = student_model, teacher_lm_head
        self.upstream = SpanCrossTokenizerKD(strategy, student_model, teacher_lm_head,
                                             student_tokenizer, teacher_tokenizer, **kwargs)
        # Preserve calculations but suppress upstream logs of response token IDs/logits.
        self.upstream._debug_count, self.upstream._span_debug_count = 3, 1
        config = getattr(student_model.model.config, 'text_config', None) or student_model.model.config
        self.softcap = getattr(config, 'final_logit_softcapping', None)
        if self.softcap is not None and self.softcap <= 0:
            raise ValueError('final_logit_softcapping must be positive')

    def _student_logits(self, hidden):
        # Match the author's actual HF student output, including final softcap.
        logits = self.student.model.lm_head(hidden, skip=False)
        return logits if self.softcap is None else torch.tanh(logits / self.softcap) * self.softcap

    def training_step(self, micro_batch):
        sm, tm = micro_batch['stu_loss_mask'].bool(), micro_batch['tea_loss_mask'].bool()
        if sm.shape[0] != tm.shape[0]:
            raise ValueError('SimCT response counts differ')
        output = self.student(micro_batch['stu_input_ids'], attention_mask=micro_batch['stu_attn_mask'],
                              allgather_logits=True, ring_attn_group=None,
                              **(micro_batch.get('stu_multi_modal_inputs') or {}))
        hidden = output['hidden_states'][-1][sm]
        th = micro_batch['teacher_hiddens']
        if isinstance(th, list):
            th = torch.cat(th, 0)
        if th.shape[0] != int(tm.sum()):
            raise ValueError('SimCT teacher hidden rows do not match loss mask')
        th = th.to(self.teacher_head.weight)
        sy = micro_batch['stu_input_ids'].roll(-1, 1)
        ty = micro_batch['tea_input_ids'].roll(-1, 1)
        total = hidden.sum() * 0.0
        aligned = response_tokens = dropped = kept = 0
        so = to = 0
        for row in range(sm.shape[0]):
            sid, tid = sy[row][sm[row]].tolist(), ty[row][tm[row]].tolist()
            sh, teacher_hidden = hidden[so:so + len(sid)], th[to:to + len(tid)]
            so, to = so + len(sid), to + len(tid)
            response_tokens += max(len(sid), len(tid))
            segments, tid, sid = self.upstream._align_sequences_with_spans(tid, sid)
            if not segments:
                continue
            # One response at a time; preserve the author's virtual-vocabulary math.
            # Target memory/throughput is not qualified by the CPU parity tests.
            sl = checkpoint(self._student_logits, sh, use_reentrant=False) if torch.is_grad_enabled() else self._student_logits(sh)
            with torch.no_grad():
                tl = self.teacher_head(teacher_hidden)  # author's teacher head has no softcap
            sv, tv = self.upstream._build_virtual_vocab_logits(segments, sl, tl, sid, tid)
            losses = self.upstream.loss_fn(sv, tv.detach(), reduction='none')
            threshold = self.args.kd.simct_gh_mask_threshold
            if threshold > 0:
                mass = self.upstream._compute_segment_teacher_Z(segments, tl, tid)
                keep = mass >= 1 / threshold
                kept += int(keep.sum())
                dropped += keep.numel() - int(keep.sum())
                losses = losses * keep.to(losses.dtype)
            total = total + losses.sum()
            aligned += sum(max(te - ts, se - ss) for ts, te, ss, se in segments)
        denominator = micro_batch['avg_micro_batch_token_num']
        if float(denominator) <= 0:
            raise ValueError('SimCT requires positive shared token normalization')
        loss = total / denominator
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('SimCT produced non-finite loss')
        return {'train/loss': loss, 'train/kd_loss': loss,
                'distill/alignment_ratio': loss.detach().new_tensor(aligned / max(response_tokens, 1)),
                'distill/simct_gh_mask_ratio': loss.detach().new_tensor(dropped / max(dropped + kept, 1))}
