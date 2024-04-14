from enum import Enum


class TransferabilityMetric(Enum):
    LOG_ME = "LogME"
    NLEEP = "NLEEP"
    PARC = "PARC"
    LFC = "LFC"
    PAC_TRAN = "PacTran"


class TransferabilityDistanceFunction(Enum):
    EUCLIDIAN = "euclidian"
    COSINE = "cosine"
    CORRELATION = "correlation"
    DOT = "dot"
