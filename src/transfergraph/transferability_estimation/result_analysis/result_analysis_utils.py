import numpy as np
from scipy.stats import stats

from transfergraph.transferability_estimation.correlation_utils import TransferabilityCorrelationMetric


def compute_correlation(
        actual_performances,
        transferability_scores,
        metric: TransferabilityCorrelationMetric,
        transferability_scores_higher_is_better=True,
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
    elif metric == TransferabilityCorrelationMetric.TOP_1:
        return top_accuracy(actual_performances, transferability_scores, 1, transferability_scores_higher_is_better)
    elif metric == TransferabilityCorrelationMetric.TOP_3:
        return top_accuracy(actual_performances, transferability_scores, 3, transferability_scores_higher_is_better)
    elif metric == TransferabilityCorrelationMetric.RELATIVE_TOP_1:
        return relative_top_accuracy(actual_performances, transferability_scores, 1, transferability_scores_higher_is_better)
    elif metric == TransferabilityCorrelationMetric.RANDOM_RELATIVE_TOP_1_ERROR:
        return random_relative_top_accuracy_error(actual_performances, transferability_scores, 1, transferability_scores_higher_is_better)
    elif metric == TransferabilityCorrelationMetric.RANDOM_RELATIVE_TOP_3_ERROR:
        return random_relative_top_accuracy_error(actual_performances, transferability_scores, 3, transferability_scores_higher_is_better)
    elif metric == TransferabilityCorrelationMetric.RANDOM_ABSOLUTE_TOP_1_ERROR:
        return random_absolute_top_accuracy_error(actual_performances, transferability_scores, 1, transferability_scores_higher_is_better)
    elif metric == TransferabilityCorrelationMetric.RANDOM_ABSOLUTE_TOP_3_ERROR:
        return random_absolute_top_accuracy_error(actual_performances, transferability_scores, 3, transferability_scores_higher_is_better)
    elif metric == TransferabilityCorrelationMetric.PERCENTILE_TOP_1:
        return percentile_of_top_k_performance(actual_performances, transferability_scores, 1, transferability_scores_higher_is_better)
    elif metric == TransferabilityCorrelationMetric.PERCENTILE_TOP_3:
        return percentile_of_top_k_performance(actual_performances, transferability_scores, 3, transferability_scores_higher_is_better)
    else:
        raise Exception(f"Unexpected TransferabilityCorrelationMetric: {metric.value}")


def top_accuracy(
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

    return best_perf_from_top_k

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


def random_relative_top_accuracy_error(
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
    mean_perf = np.mean(actual_performances)

    return (best_perf_from_top_k - mean_perf) / (best_perf - mean_perf)


def random_absolute_top_accuracy_error(
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
    mean_perf = np.mean(actual_performances)

    mean_error = best_perf - mean_perf
    predicted_error = best_perf - best_perf_from_top_k

    return mean_error - predicted_error


def percentile_of_top_k_performance(actual_performances, transferability_scores, k, transferability_scores_higher_is_better=True):
    """
    Calculate the percentile of the best performance among the top k models.

    Parameters:
    - actual_performances (list): List of actual performances of all models.
    - transferability_scores (list): Scores indicating the transferability of each model.
    - k (int): Number of top models to consider.
    - transferability_scores_higher_is_better (bool): Indicates if higher scores mean better transferability.

    Returns:
    - float: The percentile of the best performing model among the top k models in the actual performances.
    """
    # Sort indices based on transferability scores
    if transferability_scores_higher_is_better:
        top_k_indices = np.argsort(transferability_scores)[-k:][::-1]
    else:
        top_k_indices = np.argsort(transferability_scores)[:k]

    # Convert actual performances to a numpy array
    actual_performances = np.array(actual_performances)

    # Find the best performance among the top k models
    best_perf_from_top_k = np.max(actual_performances[top_k_indices])

    # Calculate the percentile of the best performance from the top k models
    percentile = stats.percentileofscore(actual_performances, best_perf_from_top_k)

    return percentile
