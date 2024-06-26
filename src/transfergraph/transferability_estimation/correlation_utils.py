from enum import Enum


class TransferabilityCorrelationMetric(Enum):
    PEARSON = 'pearson'
    SPEARMAN = 'spearman'
    KENDALL = 'kendall'
    WEIGHTED_KENDALL = 'w-kendall'
    TOP_1 = 'top@1'
    TOP_2 = 'top@2'
    TOP_3 = 'top@3'
    TOP_4 = 'top@4'
    TOP_5 = 'top@5'
    RELATIVE_TOP_1 = 'rel@1'
    RELATIVE_TOP_2 = 'rel@2'
    RELATIVE_TOP_3 = 'rel@3'
    RELATIVE_TOP_4 = 'rel@4'
    RELATIVE_TOP_5 = 'rel@5'
    PERCENTILE_TOP_1 = 'p@1'
    PERCENTILE_TOP_3 = 'p@3'
    PERCENTILE_TOP_5 = 'p@5'
    RANDOM_RELATIVE_TOP_1 = 'rrel@1'
    RANDOM_RELATIVE_TOP_3 = 'rrel@3'
    RANDOM_RELATIVE_TOP_5 = 'rrel@5'