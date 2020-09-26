import numpy as np


def random_init(D1, D2, rank):
    W1 = np.random.randn(D1.shape[0], rank) * np.sqrt(D1.mean() / rank)
    W2 = np.random.randn(D2.shape[0], rank) * np.sqrt(D2.mean() / rank)
    H = np.random.randn(rank, D1.shape[1]) * \
        np.sqrt((D1.mean() + D2.mean()) / (2 * rank))
    W1 = np.abs(W1)
    W2 = np.abs(W2)
    H = np.abs(H)
    return W1, W2, H
