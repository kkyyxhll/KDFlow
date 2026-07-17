# Evaluation

KDFlow supports evaluation during SFT, off-policy KD, and on-policy KD training. Configure an evaluation dataset and a positive evaluation interval:

```bash
--eval_dataset_path /path/to/eval_data \
--eval_split eval \
--eval_steps 100
```

With a positive `eval_steps`, evaluation runs once before a fresh training run at global step 0 and then every `eval_steps` global steps. Metrics are printed to the console and, when enabled, logged to W&B under the `eval/*` namespace.

## Evaluation modes

| Training mode | Evaluation behavior |
|---|---|
| SFT | Computes the student loss and related training metrics on the evaluation set without updating model parameters. |
| Off-policy KD | Runs teacher and student forward passes on the static evaluation samples, then reports KD and distillation metrics. |
| On-policy KD | Generates one response per evaluation prompt, runs teacher and student forward passes on the generated responses, and reports rollout and distillation metrics. |

For off-policy KD, `eval_steps` must be no smaller than `teacher_forward_n_batches` and must be a multiple of it.

## Dataset requirements

The evaluation dataset uses the same preprocessing configuration as the training dataset, including `input_key`, `output_key`, `label_key`, `image_key`, chat template, and length limits. Its schema must therefore be compatible with the training dataset.

`eval_split` defaults to `eval`. Set it explicitly when the dataset uses another split name, such as `validation` or `test`.

```bash
--eval_dataset_path openai/gsm8k \
--eval_split test \
--input_key question \
--label_key answer
```

`max_samples` also limits the number of evaluation samples. Small evaluation datasets automatically use fewer preprocessing workers to avoid multiprocessing overhead.

## On-policy generation

On-policy evaluation reuses the rollout generation settings, including `generate_max_len` and `top_p`, with the following differences:

- `temperature` is set to `0.0` for deterministic generation.
- Exactly one response is generated per prompt, regardless of `n_samples_per_prompt`.
- Generated records are saved to `SAVE_PATH/rollout_data/val/<global_step>.jsonl`.

Each record contains the prompt and generated output, plus the label when `label_key` is configured.

## Custom on-policy metrics

On-policy KD supports a custom Python evaluation function through `custom_eval_fn`. The Python file must define:

```python
def eval_fn(predictions, labels):
    return {"accuracy": 0.0}
```

- `predictions` is a list of generated response strings.
- `labels` is a list populated from the dataset field specified by `label_key`.
- The function must return a dictionary whose values can be converted to scalars.
- Metric names are automatically logged under `eval/*`; for example, `accuracy` becomes `eval/accuracy`.

A GSM8K exact-match example is provided at `examples/evaluation/gsm_8k.py`:

```bash
python -m kdflow.cli.train_kd_on_policy \
    ... \
    --eval_dataset_path /path/to/gsm8k_eval \
    --eval_split test \
    --label_key answer \
    --eval_steps 100 \
    --custom_eval_fn examples/evaluation/gsm_8k.py
```

Custom evaluation functions are supported only by on-policy KD. SFT and off-policy KD ignore `custom_eval_fn` and report their built-in loss and distillation metrics.

## Key arguments

| Argument | Default | Description |
|---|---:|---|
| `eval_dataset_path` | `None` | Evaluation dataset name or local path. |
| `eval_split` | `eval` | Split to load from the evaluation dataset. |
| `eval_steps` | `-1` | Evaluation interval in global steps. A positive value enables evaluation; `-1` disables it. |
| `label_key` | `None` | Dataset field passed to an on-policy custom evaluation function as labels. |
| `custom_eval_fn` | `None` | Python file defining `eval_fn(predictions, labels)` for on-policy KD. |
