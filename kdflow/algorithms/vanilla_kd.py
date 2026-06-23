import torch
import torch.nn.functional as F

from kdflow.loss import build_loss_fn
from kdflow.algorithms import register_algorithm
from kdflow.loss.cross_entropy import compute_cross_entropy


@register_algorithm("vanilla_kd")
class VanillaKD:
    def __init__(self, strategy, student_model, teacher_lm_head, **kwargs):
        self.strategy = strategy
        self.args = strategy.args
        self.student = student_model
        self.teacher_lm_head = teacher_lm_head
        self.loss_fn = build_loss_fn(self.args.kd.kd_loss_fn, self.args)

    def compute_multi_teacher_logits(self, teacher_hiddens, teacher_loss_mask, routing_keys):
        per_sample_counts = teacher_loss_mask.sum(dim=1).tolist()
        splits = teacher_hiddens.split(per_sample_counts, dim=0)
        teacher_to_indices = {}
        for i, key in enumerate(routing_keys):
            teacher_to_indices.setdefault(key, []).append(i)

        logits_list = [None] * len(routing_keys)
        streams = {key: torch.cuda.Stream() for key in teacher_to_indices}
        for key, indices in teacher_to_indices.items():
            with torch.cuda.stream(streams[key]):
                lm_head = self.teacher_lm_head[key]
                batched = torch.cat([splits[i] for i in indices], dim=0).to(lm_head.weight)
                batched_logits = lm_head(batched)
                sizes = [splits[i].shape[0] for i in indices]
                per_sample_logits = batched_logits.split(sizes, dim=0)
                for idx, i in enumerate(indices):
                    logits_list[i] = per_sample_logits[idx]

        for s in streams.values():
            torch.cuda.current_stream().wait_stream(s)

        return torch.cat(logits_list, dim=0)

    def training_step(self, micro_batch):
        student_input_ids = micro_batch["stu_input_ids"]
        student_attn_mask = micro_batch["stu_attn_mask"]
        student_loss_mask = micro_batch["stu_loss_mask"].bool()
        teacher_input_ids = micro_batch["tea_input_ids"]
        teacher_attn_mask = micro_batch["tea_attn_mask"]
        teacher_loss_mask = micro_batch["tea_loss_mask"].bool()
        teacher_hiddens = micro_batch.get("teacher_hiddens", None)
        avg_token_num = micro_batch["avg_micro_batch_token_num"]

        assert teacher_hiddens is not None, "micro_batch must contain `teacher_hiddens` for KD"

        mm_kwargs = micro_batch.get("stu_multi_modal_inputs") or {}

        output = self.student(
            student_input_ids,
            attention_mask=student_attn_mask,
            allgather_logits=True,
            ring_attn_group=self.strategy.ring_attn_group,
            **mm_kwargs,
        )
        student_hiddens = output["hidden_states"][-1][student_loss_mask]
        del output

        if isinstance(self.teacher_lm_head, dict):  # multi-teacher distillation
            teacher_logits = self.compute_multi_teacher_logits(
                teacher_hiddens, teacher_loss_mask, micro_batch["teacher_routing_key"]
            )
        else:
            teacher_hiddens = teacher_hiddens.to(self.teacher_lm_head.weight)
            teacher_logits = self.teacher_lm_head(teacher_hiddens)
        
        student_logits = self.student.model.lm_head(student_hiddens)
        minV = min(teacher_logits.shape[-1], student_logits.shape[-1])
        teacher_logits = teacher_logits[:, :minV]
        student_logits = student_logits[:, :minV]
        if teacher_logits.shape != student_logits.shape:
            raise ValueError(f"Teacher student shape mismatch. teacher shape: {teacher_logits.shape} vs student shape: {student_logits.shape}, teacher_loss_shape: {teacher_loss_mask.sum()} vs student_loss_shape: {student_loss_mask.sum()}")
        
        kd_loss = self.loss_fn(
            student_logits, 
            teacher_logits, 
            reduction="none",
        )
        kd_loss = kd_loss.sum() / avg_token_num
        loss_info = {"loss": kd_loss, "kd_loss": kd_loss}
        
        if self.args.kd.kd_ratio < 1:
            student_label_ids = student_input_ids.roll(shifts=-1, dims=1)[student_loss_mask]
            ce_loss = compute_cross_entropy(student_logits, student_label_ids, reduction="sum") / avg_token_num
            loss = (1 - self.args.kd.kd_ratio) * ce_loss + self.args.kd.kd_ratio * kd_loss
            loss_info["loss"] = loss
            loss_info["ce_loss"] = ce_loss

        return loss_info