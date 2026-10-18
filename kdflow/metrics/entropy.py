import torch
import torch.nn.functional as F


def compute_entropy(student_logits, **kwargs):
    """Compute the entropy of a logits distribution using numerically stable formula.

    Args:
        student_logits: Tensor of shape (num_tokens, vocab_size).
        **kwargs: Unused. Accepts extra arguments for unified metric interface.

    Returns:
        A dict containing the mean entropy value.
    """
    with torch.no_grad():
        probs = F.softmax(student_logits, dim=-1)
        entropy = torch.logsumexp(student_logits, dim=-1) - torch.sum(probs * student_logits, dim=-1)
    return {"distill/student_entropy": entropy.mean()}
