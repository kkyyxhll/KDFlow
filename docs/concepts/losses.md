# KD Loss Functions

A KD algorithm is responsible for *producing* student logits and teacher
logits; the **divergence** between them is selected by `--kd_loss_fn`.
KDFlow registers nine losses in `kdflow/loss/`, all populated through
`@register_loss(name)`:

```python
from kdflow.loss import LOSS_DICT
print(list(LOSS_DICT))
# ['kl', 'rkl', 'jsd', 'akl', 'skl', 'srkl', 'tvd', 'hrl', 'top1_ce']
```

| `--kd_loss_fn` | Name                          | File                                       | Notes                                                    |
|----------------|-------------------------------|--------------------------------------------|----------------------------------------------------------|
| `kl`           | KL divergence                 | `kdflow/loss/kl_div.py`                    | `torch.compile`d; the canonical KD loss.                 |
| `rkl`          | Reverse KL                    | `kdflow/loss/reverse_kl_div.py`            | Mode-seeking; popular on-policy choice.                  |
| `jsd`          | Jensen-Shannon divergence     | `kdflow/loss/js_div.py`                    | Symmetric; tunable mix via `--jsd_beta`.                 |
| `akl`          | Adaptive KL divergence        | `kdflow/loss/adaptive_kl_div.py`           | Mixes forward/reverse, controlled by `--adaptive_alpha`. |
| `skl`          | Skewed KL                     | `kdflow/loss/skewed_kl_div.py`             | Skewed forward KL, controlled by `--skew_lambda`.        |
| `srkl`         | Skewed reverse KL             | `kdflow/loss/skewed_rkl_div.py`            | Skewed reverse KL, controlled by `--skew_lambda`.        |
| `tvd`          | Total variation distance      | `kdflow/loss/tvd.py`                       | Bounded distance between distributions.                  |
| `hrl`          | Hierarchical Ranking Loss     | `kdflow/loss/hierarchical_ranking_loss.py` | Top-`--hrl_topk` ranking based loss.                     |
| `top1_ce`      | Top-1 cross-entropy           | `kdflow/loss/top1_ce.py`                   | CE against the teacher's argmax (cheap, hard-label KD).  |

## Quick guide for picking a loss

- **`kl`** — the safe default for off-policy KD.
- **`rkl`** — strong choice for on-policy / self-distillation (mode-seeking).
- **`akl`** — when you want adaptive mixing of forward / reverse KL.
- **`jsd`** — symmetric, robust to teacher errors.
- **`skl` / `srkl`** — skewed variants when the teacher is much stronger.
- **`tvd`** — when you need a bounded distance.
- **`hrl`** — focus on relative top-k ranking rather than full distribution.
- **`top1_ce`** — cheapest, falls back to hard-label KD.

## Hyperparameters

A few losses have their own dials, settable on the CLI:

```bash
# Jensen-Shannon
--jsd_beta 0.5                 # mixture weight

# Skewed KL / RKL
--skew_lambda 0.1

# Adaptive KL
--adaptive_alpha 0.5

# Hierarchical Ranking Loss
--hrl_topk 5
```

## Implementing your own loss

Add a file under `kdflow/loss/` and register it; see
[Extending KDFlow](../reference/extending.md). The `__init__.py` of
`kdflow.loss` auto-imports every sibling module, so your file is enough.

## See also

- [KD Algorithms](algorithms.md) — algorithms decide which logits to compare.
- [Arguments → Distillation Arguments](../reference/arguments.md#distillation-arguments) —
  the full list of distillation knobs.
