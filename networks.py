import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

class CriticNetwork(Model):
    def __init__(self, hidden_size=256, name='critic'):
        super().__init__()
        self.model_name = name
        self.checkpoint_file = self.model_name + ".h5"
        
        self.dense1 = Dense(hidden_size, activation='swish')
        self.dense2 = Dense(hidden_size/2, activation='swish')
        self.dense3 = Dense(hidden_size/4, activation='swish')
        self.q = Dense(1)
        
    def call(self, state, action):
        x = self.dense1(tf.concat([state, action], axis=1))
        x = self.dense2(x)
        x = self.dense3(x)
        q = self.q(x)
        return q
    
class ActorNetwork(Model):
    def __init__(self, n_actions, hidden_size=256, name='actor'):
        super().__init__()
        self.model_name = name
        self.checkpoint_file = self.model_name + ".h5"
        
        self.dense1 = Dense(hidden_size, activation='swish')
        self.dense2 = Dense(hidden_size*2, activation='swish')
        self.dense3 = Dense(hidden_size/2, activation='swish')
        self.actor = Dense(n_actions, activation='softmax')
        
    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        x = self.dense3(x)
        actor = self.actor(x)
        return actor
    
if __name__ == '__main__':
    from gomoku import Gomoku
    from agent import Agent
    env = Gomoku(rows=10, cols=10, n_to_win=5)
    agent = Agent(alpha=0.001, beta=0.001)
    
    state = env.get_state()
    action, _ = agent.choose_action(state)
    reward, done = env.step(action)
    next_state = env.get_state()
    # print(state)
    # print(action)
    print(reward)
    print(done)
    print(next_state)