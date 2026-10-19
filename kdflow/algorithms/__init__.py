"""Algorithms included in the submission."""

ALGO_DICT = {}


def register_algorithm(name):
    def decorator(cls):
        ALGO_DICT[name] = cls
        return cls
    return decorator


from kdflow.algorithms.simple_ctkd import SimpleCrossTokenizerKD
from kdflow.algorithms.byte_align_ctkd import ByteAlignCrossTokenizerKD
from kdflow.algorithms.h1b_span import H1bSpanCTKD
from kdflow.algorithms.topk_byte_align_ctkd import TopKByteAlignCrossTokenizerKD
from kdflow.algorithms.simct import SimCT
from kdflow.algorithms.simct_paper import SimCTPaper
from kdflow.algorithms.mp_opd_trust import TrustRMPOnPolicyKD, TrustBMPOnPolicyKD, TrustPMPOnPolicyKD
from kdflow.algorithms.mp_opd_align_new import AlignNewMPOnPolicyKD
from kdflow.algorithms.mp_opd import (
    AtomicMPOnPolicyKD, FixedMPOnPolicyKD, RandomMPOnPolicyKD,
    GrassChunkMPOnPolicyKD, AlignMPOnPolicyKD,
)

__all__ = ["ALGO_DICT", "SimpleCrossTokenizerKD", "ByteAlignCrossTokenizerKD", "TopKByteAlignCrossTokenizerKD"]
