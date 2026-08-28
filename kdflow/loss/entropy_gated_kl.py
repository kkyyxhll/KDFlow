import torch

from kdflow.loss import register_loss


@register_loss("entropy_gated_kl")
@torch.compile()
def compute_entropy_gated_kl(
    student_logits,
    teacher_logits,
    entropy_tau=0.8,
    temperature=1.0,
    reduction="none",
    **kwargs
):
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature
    student_log_probs = torch.log_softmax(student_logits, -1, dtype=torch.float32)
    teacher_log_probs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32)
    teacher_probs = teacher_log_probs.exp()
    teacher_entropy = -(teacher_probs * teacher_log_probs).sum(-1)
    gate = (teacher_entropy >= entropy_tau).to(student_log_probs.dtype)
    fkl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(-1)
    rkl = (student_log_probs.exp() * (student_log_probs - teacher_log_probs)).sum(-1)
    loss = gate * fkl + (1 - gate) * rkl

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss
