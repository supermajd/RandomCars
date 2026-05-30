"""report.py: Pretty-prints training metrics and compares against the current approved model."""

__author__ = 'Majd Jamal'

import json

from ml.config import APPROVED_DIR

# ANSI colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
GREY = '\033[90m'
BOLD = '\033[1m'
RESET = '\033[0m'

# Metrics where lower is better
LOWER_IS_BETTER = {'mae', 'rmse'}

# Metrics where higher is better
HIGHER_IS_BETTER = {'r2'}


def load_approved_metrics():
    """Loads metrics from the current approved model, if one exists.
    :return metrics: Dict of metrics from approved model, or None if missing
    """

    if not APPROVED_DIR.exists():
        return None

    meta_files = sorted(APPROVED_DIR.glob('*.metadata.json'))

    if not meta_files:
        return None

    # Take the most recent approved metadata
    latest = meta_files[-1]

    with open(latest) as f:
        meta = json.load(f)

    return {
        'model_id': meta.get('model_id', 'unknown'),
        'metrics': meta.get('metrics', {}),
    }


def format_diff(metric_name, new_value, old_value) -> str:
    """Returns a colored difference string for a single metric.
    :param metric_name: Metric key (e.g. 'mae', 'rmse', 'r2')
    :param new_value: Value from the new candidate
    :param old_value: Value from the current approved model
    :return diff_str: Colored, formatted difference string
    """

    if old_value is None or old_value == 0:
        return f'{GREY}n/a{RESET}'

    diff = new_value - old_value
    pct = (diff / abs(old_value)) * 100

    name = metric_name.lower()

    if name in LOWER_IS_BETTER:
        improved = diff < 0
    elif name in HIGHER_IS_BETTER:
        improved = diff > 0
    else:
        improved = None

    sign = '+' if diff >= 0 else ''
    text = f'{sign}{diff:.4f} ({sign}{pct:.2f}%)'

    if improved is True:
        return f'{GREEN}{text}{RESET}'
    elif improved is False:
        return f'{RED}{text}{RESET}'
    else:
        return text


def print_report(metrics, model_name) -> None:
    """Prints a formatted training report and comparison against approved model.
    :param metrics: Metrics dict from the new candidate
    :param model_name: Identifier of the new candidate model
    """

    print(f'\n{BOLD}=-=-=-=- Training Report -=-=-=-={RESET}')
    print(f'Model: {model_name}\n')

    approved = load_approved_metrics()

    if approved is None:
        print(f'{GREY}No approved model found for comparison.{RESET}\n')
        print(f'{"Metric":<10} {"Candidate":>14}')
        print('-' * 26)

        for key, value in metrics.items():
            print(f'{key.upper():<10} {value:>14.4f}')

        print()
        return

    old_metrics = approved['metrics']

    print(f'Compared to approved: {approved["model_id"]}\n')
    print(f'{"Metric":<10} {"Candidate":>14} {"Approved":>14}   {"Diff":<24}')
    print('-' * 70)

    for key, new_value in metrics.items():
        old_value = old_metrics.get(key)
        old_str = f'{old_value:.4f}' if old_value is not None else 'n/a'
        diff_str = format_diff(key, new_value, old_value)

        print(f'{key.upper():<10} {new_value:>14.4f} {old_str:>14}   {diff_str}')

    print()
