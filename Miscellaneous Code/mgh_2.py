#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 20:16:31 2019

@author: harshparikh
"""

import sys
import os
#!{sys.executable} -m pip install keras-rl

import numpy as np
import pickle

import gym
from gym import spaces
from gym.utils import seeding

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

import pickle

datafile = open('../IICdata/preprocessed_dataset_all.pickle','rb')
data = pickle.load( datafile , encoding='latin')

action_dim = 34
state_dim = 5
log = open('log_mgh_2.log','w')

def model_transition(state_dim,action_dim):
    M = np.random.normal(0,5,size=(state_dim,state_dim+action_dim))
    return M

def model_opt_action(state_dim,action_dim):
    M = np.random.normal(0,5,size=(action_dim,state_dim+action_dim))
    return M

