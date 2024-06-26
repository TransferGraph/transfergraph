from enum import Enum


class TransferabilityMethod(Enum):
    LOG_ME = "LogME"
    NLEEP = "NLEEP"
    PARC = "PARC"
    LFC = "LFC"
    PAC_TRAN = "PacTran"
    H_SCORE = "H-Score"
    REG_H_SCORE = "Reg-H-Score"

class TransferabilityDistanceFunction(Enum):
    EUCLIDIAN = "euclidian"
    COSINE = "cosine"
    CORRELATION = "correlation"
    DOT = "dot"
