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
    RANDOM_RELATIVE_TOP_1_ERROR = 'rrel@1'
    RANDOM_RELATIVE_TOP_3_ERROR = 'rrel@3'
    RANDOM_RELATIVE_TOP_5_ERROR = 'rrel@5'
    RANDOM_ABSOLUTE_TOP_1_ERROR = 'rabs@1'
    RANDOM_ABSOLUTE_TOP_3_ERROR = 'rabs@3'
    PERCENTILE_TOP_1 = 'p@1'
    PERCENTILE_TOP_3 = 'p@3'
