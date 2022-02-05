import numpy as np


def softmax(Qs, beta):

    Qs_val = np.array(list(Qs.values()))
    num = np.exp(Qs_val * beta)
    den = np.exp(Qs_val * beta).sum()
    return num / den
