from enum import Enum


class TransferabilityCorrelationMetric(Enum):
    PEARSON = 'pearson'
    SPEARMAN = 'spearman'
    KENDALL = 'kendall'
    WEIGHTED_KENDALL = 'w-kendall'
    TOP_1 = 'top@1'
    TOP_3 = 'top@3'
    TOP_5 = 'top@5'
    RELATIVE_TOP_1 = 'rel@1'
    RELATIVE_TOP_3 = 'rel@3'
    RELATIVE_TOP_5 = 'rel@5'
    PERCENTILE_TOP_1 = 'p@1'
    PERCENTILE_TOP_3 = 'p@3'
    PERCENTILE_TOP_5 = 'p@5'
