from dataclasses import dataclass, field


@dataclass
class EvalArguments:
    """Arguments for evaluation (on-policy distillation)."""

    eval_n_samples_per_prompt: int = field(
        default=1,
        metadata={"help": "Sample n responses per prompt during evaluation."}
    )
    eval_generate_max_len: int = field(
        default=None,
        metadata={"help": "Max generation tokens during evaluation. Defaults to generate_max_len."}
    )
    eval_temperature: float = field(
        default=None,
        metadata={"help": "Temperature during evaluation. Only used when explicitly set; "
                          "otherwise the rollout engine default is used."}
    )
    eval_top_p: float = field(
        default=None,
        metadata={"help": "Top-p sampling during evaluation. Only used when explicitly set; "
                          "otherwise the rollout engine default is used."}
    )
