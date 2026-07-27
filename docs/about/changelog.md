# Changelog

This page mirrors the **News** section of the project README.

## 2026/07 — KDFlow v0.2.1

🎉 KDFlow v0.2.1 focuses on **correctness**, **training throughput**,
**evaluation**, and **observability** across SFT, off-policy KD, and on-policy
KD workflows.

### Correctness and reliability

- The teacher SGLang engine now receives pre-tokenized `input_ids` instead of
  raw text, avoiding duplicate tokenization and keeping teacher hidden states
  exactly aligned with the loss mask.
- Dataset schemas are validated before preprocessing, with clear errors for
  missing required fields or columns.

### Performance

- Rollout concurrency now scales with the number of rollout engines, improving
  rollout throughput by approximately **30%–40%**.
- On-policy dynamic batching is applied before teacher forward to avoid
  repeatedly copying teacher hidden states.
- Dataset preprocessing is faster, especially for smaller datasets.
- Teacher hidden states are streamed back asynchronously to reduce transfer
  overhead.

### Evaluation and logging

- Periodic [evaluation](../user_guide/evaluation.md) is now supported for SFT,
  off-policy KD, and on-policy KD, including custom evaluation functions for
  on-policy workflows.
- End-to-end step-time logging has been added to all training workflows.
- Metrics now use structured `train/*`, `distill/*`, `rollout/*`, `timing/*`,
  and `eval/*` namespaces. Existing dashboards that depend on previous metric
  names may need to be updated.

## 2026/07 — KDFlow v0.2.0

🎉 KDFlow v0.2.0 has been released. This release brings together recent upgrades
including **chunked loss**, **multi-teacher distillation**,
**EMA teacher update** for on-policy self-distillation, **dynamic batch size**, 
and the recommended Docker image based on **sglang 0.5.12 + CUDA 12.9**.

## 2026/06 — Multi-teacher distillation

🧑‍🏫 Support **multi-teacher distillation** for both off-policy and on-policy KD.
Use `--multi_teacher_config` to provide a JSON mapping from routing keys to
teacher model paths, and `--teacher_routing_key` to read the per-sample routing
field from the dataset. See [Multi-Teacher KD](../user_guide/multi_teacher_kd.md)
for examples.

## 2026/06 — New Docker image (sglang 0.5.12 + CUDA 12.9)

🐳 A new Docker image **`kdflow:sgl0512-torch211-cu129`** based on
**sglang 0.5.12** and **CUDA 12.9** is now available on
[Docker Hub](https://hub.docker.com/repository/docker/songmzhang/kdflow/tags).
**This image is recommended going forward** — it picks up the VLM fixes from sglang ≥ 0.5.10.

## 2026/05 — EMA teacher update

🪄 Support **EMA teacher update** for on-policy self-distillation. Enable it
with `--use_ema_teacher True` and tune the decay via `--teacher_ema_decay`
(default `0.999`). When enabled, the teacher is synced from a CPU-resident
exponential moving average of the student's parameters instead of the live
student weights, giving a smoother target — see
[On-Policy KD → Self-distillation](../user_guide/on_policy_kd.md#self-distillation-teacher-from-student-weight-sync)
for the formula and tuning tips.

## 2026/04 — Dynamic batch size

⚡ Support dynamic batch size (enabled via `--use_dynamic_bsz True` and
`--max_token_len_per_gpu <N>`), which accelerates training by almost
**60 % to 100 %**.

## 2026/04 — KDFlow v0.1.3

🎉 KDFlow v0.1.3 has been released. It now supports **weight synchronisation
from student to teacher in on-policy self-distillation**, controlled by
`--teacher_update_freq` (defaults to `1`, i.e. the teacher is synced every
global step when student and teacher share the same model path).

## 2026/04 — Docker image

🐳 The Docker image for KDFlow is available on
[Docker Hub](https://hub.docker.com/repository/docker/songmzhang/kdflow/tags),
and the corresponding Dockerfile is provided in `docker/`.

## 2026/03 — KDFlow v0.1.2

🎉 KDFlow v0.1.2 has been released, supporting **multi-node TP/PP** for
extremely large teacher models (200B+).

## 2026/03 — WeChat group

💬 A KDFlow WeChat group has been created — see the project README for the QR
code.

## 2026/03 — KDFlow v0.1.1

🎉 KDFlow v0.1.1 released! Now supports **vision-language (multimodal)
models** and the **Qwen3.5 series**.
