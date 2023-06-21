import numpy as np
from agent import Agent
from gomoku import Gomoku

from agent import Agent

def predict(agent, state):
    action1d, _ = agent.choose_action(state)
    action = agent.index_1D_to_2D(int(action1d))
    return action