import numpy as np
from agentThuy import Agent as agThuy
from gomoku import Gomoku
import tensorflow as tf    
from tensorflow import keras
from agent import Agent
from predict import predict
env = Gomoku(rows=10, cols=10, n_to_win=5)
agent = agThuy()
agent.load_models()
state = env.get_state()
print(predict(agent, state))