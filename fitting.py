

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

        self.n_actions = len(df['action'].unique())
        self.stims = list(df['stimulus'].unique())
        self.n_stims = list(self.stims)


        def 
        
        