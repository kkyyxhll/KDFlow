# Changelog

## 2026/05 — EMA teacher update

Support EMA teacher update for on-policy self-distillation. Now you can enable it via `--use_ema_teacher True` and
`--teacher_ema_decay 0.999`.

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
