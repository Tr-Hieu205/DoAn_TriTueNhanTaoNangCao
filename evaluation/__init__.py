"""
evaluation package
Comparative evaluation and reporting artifacts for Vietnamese Student Feedback Classification.
"""

from evaluation.evaluate import (
    compute_comparisons,
    evaluate_baseline,
    evaluate_phobert,
    format_terminal_table,
    generate_json_report,
    generate_markdown_report,
    parse_args,
    run_evaluation,
)

__all__ = [
    "evaluate_baseline",
    "evaluate_phobert",
    "compute_comparisons",
    "format_terminal_table",
    "generate_markdown_report",
    "generate_json_report",
    "parse_args",
    "run_evaluation",
]
