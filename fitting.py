

import numpy as np
import pandas as pd
from utils import softmax
from scipy.optimize import minimize
from matplotlib import cm
import matplotlib.pyplot as plt


class ML(object):
    def __init__(self, df):
        """
            df must contains "stimulus", "action", "reward"
        """

        self.df = df

        self.n_actions = len(df['action'].unique())
        self.stims = list(df['stimulus'].unique())
        self.n_stims = list(self.stims)

    def neg_log_likelihood(self, params):

        df = self.df
        alpha = params[0]
        beta = params[1]
        bias = params[2]
        pav = params[3]
        noise = params[4]

        actions, rewards, stimuli = df['action'].values, df['reward'].values, df['stimulus'].values

        Q = {}
        V = {}
        for stim in self.stims:
            V[stim] = 0
            Q[stim] = {}
            for act in range(self.n_actions):
                Q[stim][act] = 0

        prob_log = 0
        for action, reward, stimulus in zip(actions, rewards, stimuli):

            error_Q = beta*reward - Q[stimulus][action]
            error_V = beta*reward - V[stimulus]

            q = pd.Series(Q[stimulus])
            q['go'] = q['go'] + bias + pav * V[stimulus]

            q_max = max(q)
            la = (q - q_max) - np.log(np.sum(np.exp(q - q_max)))
            p = np.exp(la)
            p_with_noise = p*(1-noise) + 1/2*noise
            prob_log = prob_log + np.log(p_with_noise[action])

            Q[stimulus][action] = Q[stimulus][action] + alpha * error_Q
            V[stimulus] = V[stimulus] + alpha * error_V

        return -prob_log
