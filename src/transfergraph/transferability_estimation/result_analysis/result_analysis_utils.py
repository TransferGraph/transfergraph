import numpy as np
from scipy.stats import stats

from transfergraph.transferability_estimation.correlation_utils import TransferabilityCorrelationMetric


def compute_correlation(
        actual_performances,
        transferability_scores,
        metric: TransferabilityCorrelationMetric,
        transferability_scores_higher_is_better=True,
        k=1,
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
        return relative_top_accuracy(actual_performances, transferability_scores, k, transferability_scores_higher_is_better)
    else:
        raise Exception(f"Unexpected TransferabilityCorrelationMetric: {metric.value}")


def relative_top_accuracy(
        actual_performances,
        transferability_scores,
        k,
        transferability_scores_higher_is_better=True,
):
    """Returns the accuracy ratio between the best model and the best among the selected top k models."""
    if transferability_scores_higher_is_better:
        top_k_indices = np.argsort(transferability_scores)[-k:][::-1]
    else:
        top_k_indices = np.argsort(transferability_scores)[:k]

    actual_performances = np.array(actual_performances)
    best_perf_from_top_k = np.max(actual_performances[top_k_indices])
    best_perf = np.max(actual_performances)

    return best_perf_from_top_k / best_perf
