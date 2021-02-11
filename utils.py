import ot
from ot.utils import clean_zeros


def compute_sinkhorn_divergence(X: np.ndarray, Y: np.ndarray, M: np.ndarray, reg=1e-3: float) -> float:
    X, Y, M = clean_zeros(X, Y, M)
    return ot.sinkhorn2(X, Y, M, reg)

def compute_emd(X: np.ndarray, Y: np.ndarray, M: np.ndarray) -> float:
    X, Y, M = clean_zeros(X, Y, M)
    return ot.emd2(X, Y, M)
