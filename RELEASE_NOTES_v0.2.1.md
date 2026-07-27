# KDFlow v0.2.1

KDFlow v0.2.1 focuses on **correctness, training throughput, evaluation, and observability** across SFT, off-policy KD, and on-policy KD workflows.

## Highlights

- Improved teacher token alignment by passing pre-tokenized `input_ids` directly to SGLang.
- Increased rollout throughput by approximately **30%–40%**.
- Added evaluation support across all training workflows.
- Reduced teacher hidden-state transfer overhead through streaming and asynchronous data movement.

## Correctness and Reliability

- Changed the teacher SGLang engine input from raw text to `input_ids`, avoiding duplicate tokenization and ensuring exact alignment between teacher hidden states and loss masks.
- Added early dataset schema validation with clear errors when required fields or dataset columns are missing.

## Performance Improvements

- Increased rollout concurrency based on the number of rollout engines, improving rollout throughput by approximately **30%–40%**.
- Moved on-policy dynamic batching before teacher forward, avoiding repeated copies of teacher hidden states.
- Accelerated dataset preprocessing, especially for smaller datasets.
- Added streaming teacher hidden-state returns and asynchronous hidden-state transfer to reduce communication overhead.

## Evaluation

- Added periodic evaluation support for:
  - Supervised Fine-Tuning
  - Off-policy Knowledge Distillation
  - On-policy Knowledge Distillation
- Added support for custom evaluation functions in on-policy KD.
- Evaluation metrics are logged under the `eval/*` namespace.

Evaluation can be enabled with:

```bash
--eval_dataset_path /path/to/eval_data \
--eval_split eval \
--eval_steps 100
```

## Logging and Observability

- Added end-to-end step-time logging for all training workflows.
- Standardized metric names using structured namespaces:
  - `train/*`
  - `distill/*`
  - `rollout/*`
  - `timing/*`
  - `eval/*`
- Added more detailed timing, rollout-length, and distillation metrics.

> Existing dashboards or scripts that depend on the previous metric names may need to be updated.

## Full Changelog

https://github.com/songmzhang/KDFlow/compare/v0.2.0...v0.2.1
