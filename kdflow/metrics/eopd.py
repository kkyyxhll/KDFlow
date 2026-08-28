import torch


def compute_eopd_metrics(student_logits, teacher_logits, entropy_tau=0.8, **kwargs):
    """Teacher entropy and forward-KL gate fraction for entropy_gated_kl."""
    with torch.no_grad():
        n = teacher_logits.shape[0]
        if n == 0:
            return {
                "distill/teacher_entropy": teacher_logits.new_zeros(()),
                "distill/fkl_gate_frac": teacher_logits.new_zeros(()),
            }
        chunk_tokens = 2048
        entropy_sum = teacher_logits.new_zeros(())
        gate_cnt = teacher_logits.new_zeros(())
        for chunk in teacher_logits.split(chunk_tokens, dim=0):
            chunk = chunk.float()
            # H = max + log(sum(exp(z - max))) - sum(softmax * z)  (numerically stable)
            logits_max = chunk.max(dim=-1, keepdim=True).values
            exp_logits = (chunk - logits_max).exp_()
            sum_exp = exp_logits.sum(dim=-1, keepdim=True)
            softmax = exp_logits.div_(sum_exp)
            entropy = (
                logits_max.squeeze(-1) + sum_exp.log().squeeze(-1)
                - (softmax * chunk).sum(dim=-1)
            )
            entropy_sum += entropy.sum()
            gate_cnt += (entropy >= entropy_tau).float().sum()
        return {
            "distill/teacher_entropy": entropy_sum / n,
            "distill/fkl_gate_frac": gate_cnt / n,
        }
