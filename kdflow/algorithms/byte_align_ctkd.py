import torch

from kdflow.algorithms import register_algorithm
from kdflow.algorithms.simple_ctkd import SimpleCrossTokenizerKD
from kdflow.token_alignment import (
    align_byte_span_batch_1to1,
    align_byte_spans_1to1,
    find_byte_overlap_token_ids,
    get_tokenizer_byte_codec,
)
from kdflow.utils.logging_utils import init_logger


logger = init_logger(__name__)


@register_algorithm("byte_align_ctkd")
class ByteAlignCrossTokenizerKD(SimpleCrossTokenizerKD):
    """Simple cross-tokenizer KD with UTF-8 byte-span 1:1 alignment."""

    def __init__(
        self,
        strategy,
        student_model,
        teacher_lm_head,
        student_tokenizer,
        teacher_tokenizer,
        **kwargs,
    ):
        # These are initialized before ``super`` because its constructor calls
        # the virtual ``_find_overlap_tokens`` method.
        self.student_byte_codec = get_tokenizer_byte_codec(student_tokenizer)
        self.teacher_byte_codec = get_tokenizer_byte_codec(teacher_tokenizer)
        super().__init__(
            strategy=strategy,
            student_model=student_model,
            teacher_lm_head=teacher_lm_head,
            student_tokenizer=student_tokenizer,
            teacher_tokenizer=teacher_tokenizer,
            **kwargs,
        )

    def _find_overlap_tokens(self):
        # Preserve the exact historical mapping for all existing ByteLevel
        # pairs. Only mixed/SentencePiece pairs use the byte-canonical map.
        if (
            self.student_byte_codec.name == "byte_level"
            and self.teacher_byte_codec.name == "byte_level"
        ):
            return super()._find_overlap_tokens()

        student_ids, teacher_ids = find_byte_overlap_token_ids(
            student_tokenizer=self.student_tokenizer,
            teacher_tokenizer=self.teacher_tokenizer,
            student_codec=self.student_byte_codec,
            teacher_codec=self.teacher_byte_codec,
        )
        device = self.teacher_lm_head.weight.device
        logger.info(
            "Num of byte-equivalent overlap tokens between student & teacher: "
            f"{len(student_ids)}"
        )
        return (
            torch.tensor(student_ids, dtype=torch.long, device=device),
            torch.tensor(teacher_ids, dtype=torch.long, device=device),
        )

    def _align_sequences(self, tea_seq, stu_seq):
        return align_byte_spans_1to1(
            teacher_tokens=tea_seq,
            student_tokens=stu_seq,
            teacher_eos_token=self.teacher_tokenizer.eos_token,
            student_eos_token=self.student_tokenizer.eos_token,
            teacher_piece_to_bytes=self.teacher_byte_codec.piece_to_bytes,
            student_piece_to_bytes=self.student_byte_codec.piece_to_bytes,
        )

    def _get_alignment_indices(
        self,
        teacher_label_ids,
        student_label_ids,
        teacher_loss_mask,
        student_loss_mask,
    ):
        """Align each micro-batch row independently, then return flat indices."""
        if teacher_label_ids.shape[0] != student_label_ids.shape[0]:
            raise ValueError(
                "Teacher and student micro-batches must contain the same number "
                "of samples for byte_align_ctkd."
            )

        teacher_batch_tokens = []
        student_batch_tokens = []
        for row_index in range(teacher_label_ids.shape[0]):
            teacher_row_ids = teacher_label_ids[row_index][
                teacher_loss_mask[row_index]
            ]
            student_row_ids = student_label_ids[row_index][
                student_loss_mask[row_index]
            ]
            teacher_row_token_ids = teacher_row_ids.cpu().tolist()
            student_row_token_ids = student_row_ids.cpu().tolist()
            teacher_batch_tokens.append(
                self.teacher_tokenizer.convert_ids_to_tokens(
                    teacher_row_token_ids
                )
            )
            student_batch_tokens.append(
                self.student_tokenizer.convert_ids_to_tokens(
                    student_row_token_ids
                )
            )

        return align_byte_span_batch_1to1(
            teacher_batch_tokens=teacher_batch_tokens,
            student_batch_tokens=student_batch_tokens,
            teacher_eos_token=self.teacher_tokenizer.eos_token,
            student_eos_token=self.student_tokenizer.eos_token,
            teacher_piece_to_bytes=self.teacher_byte_codec.piece_to_bytes,
            student_piece_to_bytes=self.student_byte_codec.piece_to_bytes,
        )
