"""Paper Eq.7-8 scores on the released SimCT candidate/alignment interface."""
import types

import torch

from . import register_algorithm
from .simct import SimCT


def paper_virtual_vocab(upstream, segments, student_logits, teacher_logits, student_ids, teacher_ids):
    # Normalize over the FULL native vocabulary at EACH autoregressive position.
    # The original builder then selects shared entries and averages span scores.
    return type(upstream)._build_virtual_vocab_logits(upstream, segments,
        torch.log_softmax(student_logits.float(), -1),
        torch.log_softmax(teacher_logits.float(), -1), student_ids, teacher_ids)


@register_algorithm('simct_paper')
class SimCTPaper(SimCT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.upstream._build_virtual_vocab_logits = types.MethodType(paper_virtual_vocab, self.upstream)
