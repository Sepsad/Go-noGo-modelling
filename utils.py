import numpy as np


def softmax(Qs_val):
    num = np.exp(Qs_val)
    den = np.exp(Qs_val).sum()
    return num / den
