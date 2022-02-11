

from statistics import mode
import numpy as np
import pandas as pd
from utils import softmax
from scipy.optimize import minimize
from matplotlib import cm
import matplotlib.pyplot as plt


class ML(object):
    def __init__(self, df, optimization_method='Nelder-Mead', model_type='RW', initial_guess=None, without_bound=True):
        """
            df must contains "stimulus", "action", "reward"
        """

        self.df = df
        self.actions = list(df['action'].unique())
        self.n_actions = len(self.actions)
        self.stims = list(df['stimulus'].unique())
        self.n_stims = list(self.stims)
        self.initial_guess = initial_guess
        self.without_bound = without_bound

        self.optimization_method = optimization_method
        self.model_type = model_type

    def neg_log_likelihood(self, params):

        df = self.df
        if (self.without_bound):
            alpha = 1/(1+np.exp(-params[0]))  # learning rate
            beta = np.exp(params[1])  # sensitivity to reward
        else:
            alpha = params[0]  # learning rate
            beta = params[1]  # sensitivity to reward

        noise = 0
        bias = 0
        pav = 0
        if self.model_type == 'RW+noise':
            if(self.without_bound):
                noise = 1/(1+np.exp(-params[2]))
            else:
                noise = params[2]  # noise
        elif self.model_type == 'RW+noise+bias':
            if(self.without_bound):
                noise = 1/(1+np.exp(-params[2]))
            else:
                noise = params[2]  # noise
            bias = params[3]  # go bias
        elif self.model_type == 'RW+noise+bias+Pav':
            if(self.without_bound):
                noise = 1/(1+np.exp(-params[2]))
                pav = np.exp(params[4])
            else:
                noise = params[2]  # noise
                pav = params[4]  # Pavlovian bias
            bias = params[3]  # go bias

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
            bounds = [(0, 1), (0, None)]
            x0 = self.initial_guess
            if(self.without_bound):
                res = minimize(self.neg_log_likelihood, x0,
                               method=self.optimization_method,
                               # bounds=bounds
                               )
            else:
                res = minimize(self.neg_log_likelihood, x0,
                               method=self.optimization_method,
                               bounds=bounds
                               )
            return res

        elif self.model_type == 'RW+noise':
            bounds = [(0, 1), (0, None), (0, 1)]
            x0 = self.initial_guess

            if(self.without_bound):
                res = minimize(self.neg_log_likelihood, x0,
                               method=self.optimization_method,
                               # bounds=bounds
                               )
            else:
                res = minimize(self.neg_log_likelihood, x0,
                               method=self.optimization_method,
                               bounds=bounds
                               )
            return res

        elif self.model_type == 'RW+noise+bias':
            bounds = [(0, 1), (0, None), (0, 1), (None, None)]
            x0 = self.initial_guess

            if(self.without_bound):
                res = minimize(self.neg_log_likelihood, x0,
                               method=self.optimization_method,
                               # bounds=bounds
                               )
            else:
                res = minimize(self.neg_log_likelihood, x0,
                               method=self.optimization_method,
                               bounds=bounds
                               )
            return res

        elif self.model_type == 'RW+noise+bias+Pav':
            bounds = [(0, 1), (0, None), (0, 1), (None, None), (0, None)]
            x0 = self.initial_guess
            if(self.without_bound):
                res = minimize(self.neg_log_likelihood, x0,
                               method=self.optimization_method,
                               # bounds=bounds
                               )
            else:
                res = minimize(self.neg_log_likelihood, x0,
                               method=self.optimization_method,
                               bounds=bounds
                               )
            return res
        else:
            raise ValueError(
                'model_type must be "RW" or "RW+noise" or "RW+noise+bias" or "RW+noise+bias+Pav"')
