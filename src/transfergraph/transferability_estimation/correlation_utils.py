from enum import Enum


class TransferabilityCorrelationMetric(Enum):
    PEARSON = 'pearson'
    SPEARMAN = 'spearman'
    KENDALL = 'kendall'
    WEIGHTED_KENDALL = 'w-kendall'
    RELATIVE_TOP_K = 'rel@k'
