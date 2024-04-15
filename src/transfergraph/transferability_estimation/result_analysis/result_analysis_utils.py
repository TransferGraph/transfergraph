import numpy as np
from scipy.stats import stats

from transfergraph.transferability_estimation.correlation_utils import TransferabilityCorrelationMetric


def compute_correlation(
        actual_performances,
        transferability_scores,
        metric: TransferabilityCorrelationMetric,
        transferability_scores_higher_is_better=True
):
    """Return a correlation score, according to the metric."""

    if metric == TransferabilityCorrelationMetric.PEARSON:
        return stats.pearsonr(actual_performances, transferability_scores)[0]
    elif metric == TransferabilityCorrelationMetric.SPEARMAN:
        return stats.spearmanr(actual_performances, transferability_scores)[0]
    elif metric == TransferabilityCorrelationMetric.KENDALL:
        return stats.kendalltau(actual_performances, transferability_scores)[0]
    elif metric == TransferabilityCorrelationMetric.WEIGHTED_KENDALL:
        return stats.weightedtau(actual_performances, transferability_scores)[0]
    elif metric == TransferabilityCorrelationMetric.RELATIVE_TOP_K:
        return relative_top_accuracy(actual_performances, transferability_scores, transferability_scores_higher_is_better)
    else:
        raise Exception(f"Unexpected TransferabilityCorrelationMetric: {metric.value}")


def relative_top_accuracy(
        actual_performances,
        transferability_scores,
        transferability_scores_higher_is_better=True
):
    """Returns the accuracy ratio between the best model and the selected one."""
    if transferability_scores_higher_is_better:
        best_from_transfer = np.argmax(transferability_scores)
    else:
        best_from_transfer = np.argmin(transferability_scores)
    best_perf_from_transfer = actual_performances[best_from_transfer]
    best_perf = np.max(actual_performances)
    return best_perf_from_transfer / best_perf
