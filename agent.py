import imp
from multiprocessing.dummy import active_children
import sys
import pandas as pd
import numpy as np
import random

from sympy import re
from utils import softmax


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
    def __init__(self, bandit, beta=0.1, alpha=0.2):
        self.Q = {}
        self.bandit = bandit
        self.beta = beta
        self.alpha = alpha
        self.actions = self.bandit.actions
        self.contexts = self.bandit.get_context_list()
        self.n = len(self.actions)
        # print(self.contexts)
        # init small random values to avoid ties
        for context in self.contexts:
            self.Q[context] = {}
            for action in self.actions:
                self.Q[context][action] = np.random.uniform(0, 1e-4)

        self.log = {'context': [], 'action': [],
                    'reward': [], 'Q(go)': [], 'Q(nogo)': []}

    def choose_action(self, context):
        p = softmax(self.Q[context], self.beta)
        action = np.random.choice(self.actions, p=p)
        return action

    def update_action_value(self, context, action, reward):
        error = reward - self.get_Q(context, action)
        self.Q[context][action] = self.Q[context][action] + self.alpha * error

    def get_Q(self, context, action):
        return self.Q[context][action]

    def run(self):
        contexts_sequence = self.bandit.get_random_context_list()

        for context in contexts_sequence:
            action = self.choose_action(context)
            reward = self.bandit.reward(action, context)

            # update action value
            self.update_action_value(context, action, reward)

            # track performance
            self.log['context'].append(context)
            self.log['action'].append(action)
            self.log['reward'].append(reward)
            self.log['Q(go)'].append(self.get_Q(context, 'go'))
            self.log['Q(nogo)'].append(self.get_Q(context, 'nogo'))


def run_experiment(bandit, n_runs, beta, alpha):
    print('Running a go-nogo experiment simulation with beta={}, alpha={}'.format(beta, alpha))

    # init agent
    agent = Agent(bandit, beta, alpha)

    for _ in range(n_runs):
        agent.run()

    df = pd.DataFrame(agent.log)

    return df
