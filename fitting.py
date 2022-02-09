

from statistics import mode
import numpy as np
import pandas as pd
from utils import softmax
from scipy.optimize import minimize
from matplotlib import cm
import matplotlib.pyplot as plt


class ML(object):
    def __init__(self, df, optimization_method='Nelder-Mead', model_type='RW'):
        """
            df must contains "stimulus", "action", "reward"
        """

        self.df = df
        self.actions = list(df['action'].unique())
        self.n_actions = len(self.actions)
        self.stims = list(df['stimulus'].unique())
        self.n_stims = list(self.stims)

        self.optimization_method = optimization_method
        self.model_type = model_type

    def neg_log_likelihood(self, params):

        df = self.df
        alpha = params[0]  # learning rate
        beta = params[1]  # sensitivity to reward
        noise = 0
        bias = 0
        pav = 0
        if self.model_type == 'RW+noise':
            noise = params[2]  # noise
        elif self.model_type == 'RW+noise+bias':
            noise = params[2]  # noise
            bias = params[3]  # go bias
        elif self.model_type == 'RW+noise+bias+Pav':
            noise = params[2]  # noise
            bias = params[3]  # go bias
            pav = params[4]  # Pavlovian bias

        actions, rewards, stimuli = df['action'].values, df['reward'].values, df['stimulus'].values

        Q = {}
        V = {}
        for stim in self.stims:
            V[stim] = 0
            Q[stim] = {}
            for act in self.actions:
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

    def fit_model(self):

        if self.model_type == 'RW':
            bounds = [(0, 1), (0, 10)]
            res = minimize(self.neg_log_likelihood, [0.1, 0.1],
                           method=self.optimization_method, bounds=bounds)
            return res
        elif self.model_type == 'RW+noise':
            bounds = [(0, 1), (0, 10), (0, 1)]
            res = minimize(self.neg_log_likelihood, [0.1, 0.1, 0.1],
                           method=self.optimization_method, bounds=bounds)
            return res
        elif self.model_type == 'RW+noise+bias':
            bounds = [(0, 1), (0, 10), (0, 1), (0, 10)]
            res = minimize(self.neg_log_likelihood, [0.1, 0.1, 0.1, 0.1],
                           method=self.optimization_method, bounds=bounds)
            return res
        elif self.model_type == 'RW+noise+bias+Pav':
            bounds = [(0, 1), (0, 10), (0, 1), (0, 10), (-0.00001, 10)]
            res = minimize(self.neg_log_likelihood, [0.1, 0.1, 0.1, 0.1, 0.1],
                           method=self.optimization_method, bounds=bounds)
            return res
        else:
            raise ValueError(
                'model_type must be "RW" or "RW+noise" or "RW+noise+bias" or "RW+noise+bias+Pav"')
