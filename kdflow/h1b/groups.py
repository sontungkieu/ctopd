"""Byte-piece groups using Strict's codecs and EOS segmentation."""
from dataclasses import dataclass
from itertools import zip_longest
from kdflow.token_alignment import _split_on_eos


@dataclass(frozen=True)
class Group:
    kind: str
    s_idx: tuple
    t_idx: tuple
    b0: int
    b1: int
    content: bytes
    eos: bool = False


@dataclass(frozen=True)
class RowGroups:
    groups: tuple
    valid: bool
    reason: str = ''
    student_tokens: int = 0
    teacher_tokens: int = 0


def align_byte_span_groups_batch(tea_tokens, stu_tokens, teacher_eos_token,
                                student_eos_token, teacher_piece_to_bytes,
                                student_piece_to_bytes):
    if len(tea_tokens) != len(stu_tokens):
        raise ValueError('H1b response counts differ')
    rows, so, to = [], 0, 0
    for ts, ss in zip(tea_tokens, stu_tokens):
        groups, reason, sc, tc = [], '', 0, 0
        tsegments, ssegments = _split_on_eos(ts, teacher_eos_token), _split_on_eos(ss, student_eos_token)
        for titem, sitem in zip_longest(tsegments, ssegments):
            if titem is None or sitem is None:
                reason = 'eos_segment_mismatch'
                break
            tseg, teos = titem
            sseg, seos = sitem
            tp = [teacher_piece_to_bytes(p) for _, p in tseg]
            sp = [student_piece_to_bytes(p) for _, p in sseg]
            tc += len(tp)
            sc += len(sp)
            if b''.join(tp) != b''.join(sp):
                reason = 'byte_mismatch'
                break
            if any(not p for p in tp + sp):
                reason = 'empty_piece'
                break
            ti = si = t0 = s0 = b0 = 0
            tend, send = [], []
            for p in tp: tend.append((tend[-1] if tend else 0) + len(p))
            for p in sp: send.append((send[-1] if send else 0) + len(p))
            while ti < len(tp) and si < len(sp):
                if tend[ti] < send[si]: ti += 1; continue
                if tend[ti] > send[si]: si += 1; continue
                tt = tuple(to + tseg[k][0] for k in range(t0, ti + 1))
                st = tuple(so + sseg[k][0] for k in range(s0, si + 1))
                groups.append(Group('strict' if len(tt) == len(st) == 1 else 'mis',
                                    st, tt, b0, tend[ti], b''.join(tp[t0:ti+1])))
                b0 = tend[ti]
                ti += 1; si += 1; t0, s0 = ti, si
            if teos is not None and seos is not None:
                groups.append(Group('strict', (so+seos,), (to+teos,), b0, b0, b'', True))
        # Content denominators count even a rejected response completely.
        sc = sum(p != student_eos_token for p in ss)
        tc = sum(p != teacher_eos_token for p in ts)
        rows.append(RowGroups(tuple(groups) if not reason else (), not reason, reason, sc, tc))
        so += len(ss); to += len(ts)
    return rows
