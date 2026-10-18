import re
from itertools import zip_longest
from typing import Callable, List, Optional, Sequence, Tuple


PieceToBytes = Callable[[str], bytes]


class TokenByteCodec:
    """Convert tokenizer vocabulary pieces back to the bytes they represent."""

    def __init__(self, name: str, piece_to_bytes: PieceToBytes):
        self.name = name
        self.piece_to_bytes = piece_to_bytes

    def __repr__(self) -> str:
        return f"TokenByteCodec(name={self.name!r})"


def is_byte_level_tokenizer(tokenizer) -> bool:
    """Return whether a fast tokenizer uses a ByteLevel pre-tokenizer/decoder."""
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        return False
    return "ByteLevel" in repr(getattr(backend, "pre_tokenizer", None)) or (
        "ByteLevel" in repr(getattr(backend, "decoder", None))
    )


def is_sentencepiece_tokenizer(tokenizer) -> bool:
    """Return whether tokenizer pieces use SentencePiece whitespace/byte syntax.

    Gemma 3's fast tokenizer is serialized as a Hugging Face BPE backend, but it
    retains SentencePiece-compatible pieces: spaces are represented by ``▁``
    and byte fallback pieces by ``<0xNN>``.  Consequently the backend model type
    alone is not a reliable discriminator.
    """
    if getattr(tokenizer, "sp_model", None) is not None:
        return True

    tokenizer_name = type(tokenizer).__name__.lower()
    if "gemmatokenizer" in tokenizer_name or "sentencepiece" in tokenizer_name:
        return True

    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        return False
    components = " ".join(
        repr(getattr(backend, name, None))
        for name in ("normalizer", "pre_tokenizer", "decoder", "model")
    )
    return "Metaspace" in components or (
        "ByteFallback" in components and "Replace" in components
    )


def _bytes_to_unicode() -> dict[int, str]:
    """Build the reversible byte-to-unicode alphabet used by ByteLevel BPE."""
    byte_values = list(range(ord("!"), ord("~") + 1))
    byte_values += list(range(ord("¡"), ord("¬") + 1))
    byte_values += list(range(ord("®"), ord("ÿ") + 1))
    unicode_values = byte_values[:]
    extra_index = 0
    for byte_value in range(256):
        if byte_value not in byte_values:
            byte_values.append(byte_value)
            unicode_values.append(256 + extra_index)
            extra_index += 1
    return dict(
        zip(byte_values, [chr(value) for value in unicode_values], strict=True)
    )


_BYTE_LEVEL_DECODER = {
    character: byte_value for byte_value, character in _bytes_to_unicode().items()
}


def byte_level_piece_to_bytes(piece: str) -> bytes:
    """Decode a ByteLevel vocabulary piece into the source bytes it represents."""
    output = bytearray()
    for character in piece:
        byte_value = _BYTE_LEVEL_DECODER.get(character)
        if byte_value is None:
            # Added tokens may remain literal Unicode even in a ByteLevel tokenizer.
            output.extend(character.encode("utf-8"))
        else:
            output.append(byte_value)
    return bytes(output)


_SENTENCEPIECE_BYTE_RE = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")
_UNUSED_TOKEN_RE = re.compile(r"^<unused\d+>$")


def sentencepiece_piece_to_bytes(piece: str) -> bytes:
    """Decode a SentencePiece/Gemma vocabulary piece into source bytes."""
    byte_match = _SENTENCEPIECE_BYTE_RE.fullmatch(piece)
    if byte_match is not None:
        return bytes([int(byte_match.group(1), 16)])
    return piece.replace("▁", " ").encode("utf-8")


BYTE_LEVEL_CODEC = TokenByteCodec("byte_level", byte_level_piece_to_bytes)
SENTENCEPIECE_CODEC = TokenByteCodec(
    "sentencepiece", sentencepiece_piece_to_bytes
)


def get_tokenizer_byte_codec(tokenizer) -> TokenByteCodec:
    """Return the byte codec supported by ``byte_align_ctkd`` for a tokenizer."""
    # Check ByteLevel first: class-name fallbacks for SentencePiece tokenizers
    # must never change the legacy ByteLevel path.
    if is_byte_level_tokenizer(tokenizer):
        return BYTE_LEVEL_CODEC
    if is_sentencepiece_tokenizer(tokenizer):
        return SENTENCEPIECE_CODEC
    raise ValueError(
        "byte_align_ctkd supports fast ByteLevel tokenizers and "
        "SentencePiece-compatible tokenizers (including Gemma 3); got "
        f"{type(tokenizer).__name__}"
    )


def _special_token_ids(tokenizer) -> set[int]:
    special_ids = {
        int(token_id)
        for token_id in (getattr(tokenizer, "all_special_ids", None) or [])
        if token_id is not None
    }
    added_tokens = getattr(tokenizer, "added_tokens_decoder", None) or {}
    for token_id, token in added_tokens.items():
        if getattr(token, "special", False):
            special_ids.add(int(token_id))
    return special_ids


def _unique_byte_vocab(tokenizer, codec: TokenByteCodec) -> dict[bytes, int]:
    """Build an unambiguous content-byte to token-id vocabulary."""
    special_ids = _special_token_ids(tokenizer)
    candidates: dict[bytes, set[int]] = {}
    for piece, token_id in tokenizer.get_vocab().items():
        token_id = int(token_id)
        if token_id in special_ids or _UNUSED_TOKEN_RE.fullmatch(piece):
            continue
        piece_bytes = codec.piece_to_bytes(piece)
        if not piece_bytes:
            continue
        candidates.setdefault(piece_bytes, set()).add(token_id)

    # Multiple IDs with identical surface bytes do not define a safe logit
    # correspondence, so discard collisions instead of selecting arbitrarily.
    return {
        piece_bytes: next(iter(token_ids))
        for piece_bytes, token_ids in candidates.items()
        if len(token_ids) == 1
    }


def find_byte_overlap_token_ids(
    student_tokenizer,
    teacher_tokenizer,
    student_codec: Optional[TokenByteCodec] = None,
    teacher_codec: Optional[TokenByteCodec] = None,
) -> Tuple[List[int], List[int]]:
    """Map content-equivalent vocabulary entries by decoded bytes.

    This mapping is intended only for pairs containing a SentencePiece-style
    tokenizer.  Existing ByteLevel/ByteLevel pairs keep the legacy token-string
    mapping in ``SimpleCrossTokenizerKD``.
    """
    student_codec = student_codec or get_tokenizer_byte_codec(student_tokenizer)
    teacher_codec = teacher_codec or get_tokenizer_byte_codec(teacher_tokenizer)
    student_vocab = _unique_byte_vocab(student_tokenizer, student_codec)
    teacher_vocab = _unique_byte_vocab(teacher_tokenizer, teacher_codec)
    overlap = sorted(student_vocab.keys() & teacher_vocab.keys())
    student_eos = student_tokenizer.eos_token_id
    teacher_eos = teacher_tokenizer.eos_token_id
    paired_ids = [
        (student_vocab[piece_bytes], teacher_vocab[piece_bytes])
        for piece_bytes in overlap
    ]
    paired_ids = [
        (student_id, teacher_id)
        for student_id, teacher_id in paired_ids
        if student_id != student_eos and teacher_id != teacher_eos
    ]
    student_ids = [student_id for student_id, _ in paired_ids]
    teacher_ids = [teacher_id for _, teacher_id in paired_ids]

    if student_eos is not None and teacher_eos is not None:
        student_ids.append(int(student_eos))
        teacher_ids.append(int(teacher_eos))
    return student_ids, teacher_ids


def _split_on_eos(
    tokens: Sequence[str], eos_token: Optional[str]
) -> List[Tuple[List[Tuple[int, str]], Optional[int]]]:
    """Split flattened response tokens into samples while retaining EOS indices."""
    segments = []
    current = []
    for index, token in enumerate(tokens):
        if eos_token is not None and token == eos_token:
            segments.append((current, index))
            current = []
        else:
            current.append((index, token))
    if current or not segments:
        segments.append((current, None))
    return segments


def _align_segment_1to1(
    teacher_segment: Sequence[Tuple[int, str]],
    student_segment: Sequence[Tuple[int, str]],
    teacher_piece_to_bytes: PieceToBytes = byte_level_piece_to_bytes,
    student_piece_to_bytes: PieceToBytes = byte_level_piece_to_bytes,
) -> Tuple[List[int], List[int]]:
    teacher_pieces = [
        teacher_piece_to_bytes(token) for _, token in teacher_segment
    ]
    student_pieces = [
        student_piece_to_bytes(token) for _, token in student_segment
    ]

    # Both tokenizers must describe the same completion bytes. If they do not,
    # aligning equal-length spans would silently pair unrelated content.
    if b"".join(teacher_pieces) != b"".join(student_pieces):
        return [], []

    teacher_ends = []
    student_ends = []
    cursor = 0
    for piece in teacher_pieces:
        cursor += len(piece)
        teacher_ends.append(cursor)
    cursor = 0
    for piece in student_pieces:
        cursor += len(piece)
        student_ends.append(cursor)

    teacher_aligned = []
    student_aligned = []
    teacher_index = student_index = 0
    teacher_group_start = student_group_start = 0

    while teacher_index < len(teacher_ends) and student_index < len(student_ends):
        teacher_end = teacher_ends[teacher_index]
        student_end = student_ends[student_index]
        if teacher_end < student_end:
            teacher_index += 1
            continue
        if teacher_end > student_end:
            student_index += 1
            continue

        teacher_group_size = teacher_index - teacher_group_start + 1
        student_group_size = student_index - student_group_start + 1
        if teacher_group_size == 1 and student_group_size == 1:
            teacher_piece = teacher_pieces[teacher_index]
            student_piece = student_pieces[student_index]
            if teacher_piece == student_piece and teacher_piece:
                teacher_aligned.append(teacher_segment[teacher_index][0])
                student_aligned.append(student_segment[student_index][0])

        teacher_index += 1
        student_index += 1
        teacher_group_start = teacher_index
        student_group_start = student_index

    return teacher_aligned, student_aligned


def align_byte_spans_1to1(
    teacher_tokens: Sequence[str],
    student_tokens: Sequence[str],
    teacher_eos_token: Optional[str],
    student_eos_token: Optional[str],
    teacher_piece_to_bytes: PieceToBytes = byte_level_piece_to_bytes,
    student_piece_to_bytes: PieceToBytes = byte_level_piece_to_bytes,
) -> Tuple[List[int], List[int]]:
    """Align only 1:1 decoded-byte token spans, treating EOS as a boundary.

    Content groups that are 1:N, N:1, or N:M are discarded. If both sides have
    EOS at the end of a paired sample, those EOS positions are aligned directly
    and the byte coordinate is reset for the next sample.
    """
    teacher_segments = _split_on_eos(teacher_tokens, teacher_eos_token)
    student_segments = _split_on_eos(student_tokens, student_eos_token)
    teacher_aligned = []
    student_aligned = []

    for teacher_item, student_item in zip_longest(
        teacher_segments, student_segments
    ):
        if teacher_item is None or student_item is None:
            break
        teacher_segment, teacher_eos_index = teacher_item
        student_segment, student_eos_index = student_item
        segment_teacher_aligned, segment_student_aligned = _align_segment_1to1(
            teacher_segment,
            student_segment,
            teacher_piece_to_bytes=teacher_piece_to_bytes,
            student_piece_to_bytes=student_piece_to_bytes,
        )
        teacher_aligned.extend(segment_teacher_aligned)
        student_aligned.extend(segment_student_aligned)

        if teacher_eos_index is not None and student_eos_index is not None:
            teacher_aligned.append(teacher_eos_index)
            student_aligned.append(student_eos_index)

    return teacher_aligned, student_aligned


def align_byte_span_batch_1to1(
    teacher_batch_tokens: Sequence[Sequence[str]],
    student_batch_tokens: Sequence[Sequence[str]],
    teacher_eos_token: Optional[str],
    student_eos_token: Optional[str],
    teacher_piece_to_bytes: PieceToBytes = byte_level_piece_to_bytes,
    student_piece_to_bytes: PieceToBytes = byte_level_piece_to_bytes,
) -> Tuple[List[int], List[int]]:
    """Align each sample separately and return indices into flattened tensors."""
    if len(teacher_batch_tokens) != len(student_batch_tokens):
        raise ValueError(
            "Teacher and student micro-batches must contain the same number of samples."
        )

    teacher_aligned = []
    student_aligned = []
    teacher_offset = 0
    student_offset = 0
    for teacher_tokens, student_tokens in zip(
        teacher_batch_tokens, student_batch_tokens, strict=True
    ):
        row_teacher_aligned, row_student_aligned = align_byte_spans_1to1(
            teacher_tokens=teacher_tokens,
            student_tokens=student_tokens,
            teacher_eos_token=teacher_eos_token,
            student_eos_token=student_eos_token,
            teacher_piece_to_bytes=teacher_piece_to_bytes,
            student_piece_to_bytes=student_piece_to_bytes,
        )
        teacher_aligned.extend(
            teacher_offset + index for index in row_teacher_aligned
        )
        student_aligned.extend(
            student_offset + index for index in row_student_aligned
        )
        teacher_offset += len(teacher_tokens)
        student_offset += len(student_tokens)

    return teacher_aligned, student_aligned
