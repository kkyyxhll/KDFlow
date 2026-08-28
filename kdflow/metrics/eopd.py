import torch


def compute_eopd_metrics(student_logits, teacher_logits, entropy_tau=0.8, **kwargs):
    """Teacher entropy and forward-KL gate fraction for entropy_gated_kl."""
    with torch.no_grad():
        if teacher_logits.shape[0] == 0:
            return {
                "distill/teacher_entropy": teacher_logits.new_zeros(()),
                "distill/fkl_gate_frac": teacher_logits.new_zeros(()),
            }
        teacher_log_probs = torch.log_softmax(teacher_logits.float(), -1)
        teacher_entropy = -(teacher_log_probs.exp() * teacher_log_probs).sum(-1)
        gate_frac = (teacher_entropy >= entropy_tau).float().mean()
        return {
            "distill/teacher_entropy": teacher_entropy.mean(),
            "distill/fkl_gate_frac": gate_frac,
        }
