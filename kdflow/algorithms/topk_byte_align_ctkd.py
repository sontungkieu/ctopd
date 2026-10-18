from typing import Tuple

import torch

from kdflow.algorithms import register_algorithm
from kdflow.algorithms.byte_align_ctkd import ByteAlignCrossTokenizerKD
from kdflow.metrics.entropy import compute_entropy


def select_student_topk_logits(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    topk: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select the student's top-k shared-vocabulary entries on both sides.

    The returned tensors contain matching vocabulary entries. Downstream KD
    losses apply softmax/log-softmax to these reduced logits, which
    re-normalizes both distributions over exactly this top-k support.
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            "Student and teacher shared-vocabulary logits must have the same "
            f"shape, got {student_logits.shape} and {teacher_logits.shape}."
        )
    if topk <= 0 or topk > student_logits.shape[-1]:
        raise ValueError(
            f"topk must be in [1, {student_logits.shape[-1]}], got {topk}."
        )

    student_topk_logits, topk_indices = torch.topk(
        student_logits,
        k=topk,
        dim=-1,
    )
    teacher_topk_logits = teacher_logits.gather(-1, topk_indices)
    return student_topk_logits, teacher_topk_logits, topk_indices


@register_algorithm("topk_byte_align_ctkd")
class TopKByteAlignCrossTokenizerKD(ByteAlignCrossTokenizerKD):
    """Byte-aligned CTKD restricted to each student's shared-vocab top-k."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ctkd_topk = self.args.kd.ctkd_topk
        overlap_vocab_size = self.student_overlap_token_ids.numel()
        if self.ctkd_topk < 2:
            raise ValueError(
                "ctkd_topk must be at least 2 because re-normalizing a "
                "single logit produces a degenerate distribution."
            )
        if self.ctkd_topk > overlap_vocab_size:
            raise ValueError(
                f"ctkd_topk ({self.ctkd_topk}) exceeds the shared vocabulary "
                f"size ({overlap_vocab_size})."
            )
        if (
            self.args.kd.kd_loss_fn == "hrl"
            and self.args.kd.hrl_topk > self.ctkd_topk
        ):
            raise ValueError(
                f"hrl_topk ({self.args.kd.hrl_topk}) cannot exceed ctkd_topk "
                f"({self.ctkd_topk})."
            )

        self._shared_vocab_loss_fn = self.loss_fn
        self.loss_fn = self._topk_loss
        if self.metric_fns:
            self.metric_fns = [self._topk_entropy]

    def _topk_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        student_topk_logits, teacher_topk_logits, _ = select_student_topk_logits(
            student_logits,
            teacher_logits,
            self.ctkd_topk,
        )
        return self._shared_vocab_loss_fn(
            student_topk_logits,
            teacher_topk_logits,
            **kwargs,
        )

    def _topk_entropy(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        **kwargs,
    ):
        student_topk_logits, _, _ = select_student_topk_logits(
            student_logits,
            teacher_logits,
            self.ctkd_topk,
        )
        return compute_entropy(student_logits=student_topk_logits)
