import imp
from multiprocessing.dummy import active_children
import sys
import pandas as pd
import numpy as np
import random

from utils import softmax
# from sympy import re


class Bandit(object):
    def __init__(self):
        self.contexts = {'go2win': {'reward': [1, 0], 'correct_response': 'go'},
                         'nogo2win': {'reward': [1, 0], 'correct_response': 'nogo'},
                         'go2avoidPun': {'reward': [0, -1], 'correct_response': 'go'},
                         'nogo2avoidPun': {'reward': [0, -1], 'correct_response': 'nogo'}}

        self.actions = ['go', 'nogo']
        self.n = len(self.actions)

        self.better_outcome_prob = 0.80

    def get_context_list(self):
        return list(self.contexts.keys())

    def get_random_context_list(self):
        contexts = self.get_context_list()
        return random.sample(contexts, len(contexts))

    def get_action_list(self):
        return self.actions

    def reward(self, action, context):

        if(action not in self.actions):
            print("Action not in ", self.actions)
            sys.exit(-1)
        if(context not in self.contexts):
            print("Context not in ", self.contexts)
            sys.exit(-1)

        possible_rewards = self.contexts[context]['reward']
        correct_response = self.contexts[context]['correct_response']

        if(action == correct_response):
            if(np.random.rand() < self.better_outcome_prob):
                r = max(possible_rewards)
            else:
                r = min(possible_rewards)
        else:
            if(np.random.rand() < self.better_outcome_prob):
                r = min(possible_rewards)
            else:
                r = max(possible_rewards)

        return r


class Agent(object):
    def __init__(self, bandit, params={}, beta2=False):
        self.Q = {}
        self.V = {}
        self.bandit = bandit
        self.params = params
        self.beta2 = beta2

        try:
            self.alpha = params['alpha']
            if(beta2):
                self.beta_rew = params['beta_rew']
                self.beta_pun = params['beta_pun']

            else:
                self.beta = params['beta']
            self.Pav = params['Pav']
            self.noise = params['noise']
            self.bias = params['bias']
        except:
            print("Error in parameters")
            sys.exit(-1)

        self.actions = self.bandit.actions
        self.contexts = self.bandit.get_context_list()
        self.n = len(self.actions)

        # init small random values to avoid ties
        for context in self.contexts:
            self.V[context] = np.random.uniform(0, 1e-4)
            self.Q[context] = {}
            for action in self.actions:
                self.Q[context][action] = np.random.uniform(0, 1e-4)

        self.log = {'context': [], 'action': [],
                    'reward': [], 'p_go': [], 'Q(go)': [], 'Q(nogo)': []}

    def choose_action(self, context):
        Q_go = self.get_Q(context, 'go')
        Q_nogo = self.get_Q(context, 'nogo')
        V_state = self.get_V(context)

        p = softmax([(Q_go + self.bias + self.Pav*V_state), Q_nogo]
                    ) * (1 - self.noise) + self.noise/2

        action = np.random.choice(self.actions, p=p)
        p_go = p[0]

        return action, p_go

    def update_action_value(self, context, action, reward):
        if(self.beta2):
            if ('win' in context):
                error_Q = self.beta_rew*reward - self.get_Q(context, action)
                error_V = self.beta_rew*reward - self.get_V(context)

            elif('avoidPun' in context):
                error_Q = self.beta_pun*reward - self.get_Q(context, action)
                error_V = self.beta_pun*reward - self.get_V(context)
        else:
            error_Q = self.beta*reward - self.get_Q(context, action)
            error_V = self.beta*reward - self.get_V(context)

        self.Q[context][action] = self.Q[context][action] + \
            self.alpha * error_Q
        self.V[context] = self.V[context] + self.alpha * error_V

    def get_Q(self, context, action):
        return self.Q[context][action]

    def get_V(self, context):
        return self.V[context]

    def run(self):
        contexts_sequence = self.bandit.get_random_context_list()

        for context in contexts_sequence:
            action, p_go = self.choose_action(context)
            reward = self.bandit.reward(action, context)

            # update action value
            self.update_action_value(context, action, reward)

            # track performance
            self.log['context'].append(context)
            self.log['action'].append(action)
            self.log['reward'].append(reward)
            self.log['p_go'].append(p_go)
            self.log['Q(go)'].append(self.get_Q(context, 'go'))
            self.log['Q(nogo)'].append(self.get_Q(context, 'nogo'))


def run_experiment(bandit, n_runs, params={}, beta2=False) -> pd.DataFrame:
    # print('Running a go-nogo experiment simulation with params = {}'.format(params))

    # init agent
    agent = Agent(bandit, params=params, beta2=beta2)

    for _ in range(n_runs):
        agent.run()

    df = pd.DataFrame(agent.log)

    return df
