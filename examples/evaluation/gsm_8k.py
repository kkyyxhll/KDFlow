"""GSM8K evaluation example for --custom_eval_fn."""

import re


def extract_solution(text):
    solutions = re.findall(r"#### (\-?[0-9\.\,]+)", text)
    return solutions[-1].replace(",", "") if solutions else None


def eval_fn(predictions, labels):
    """Compute verl-style strict exact-match accuracy."""
    correct = 0
    for prediction, label in zip(predictions, labels):
        answer = extract_solution(prediction)
        correct += answer is not None and answer == extract_solution(label)
    return {"accuracy": correct / max(len(labels), 1)}
