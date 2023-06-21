import numpy as np
import tensorflow as tf
from tensorflow import keras
from networks import ActorNetwork, CriticNetwork

rows = 10
cols = 10

class Agent:
    def __init__(self, alpha=0.00001, beta=0.00001,
                 gamma=0.99, n_actions=100, max_size=1000000, tau=0.005,
                 hidden_size=128, batch_size=32, noise=0.1):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.noise = noise
        
        self.actor = ActorNetwork(n_actions=n_actions, hidden_size=hidden_size, name='actor')
        self.critic = CriticNetwork(hidden_size=hidden_size, name='critic')
        self.target_actor = ActorNetwork(n_actions=n_actions, hidden_size=hidden_size, name='target_actor')
        self.target_critic = CriticNetwork(hidden_size=hidden_size, name='target_critic')
        
        self.actor.compile(optimizer=keras.optimizers.RMSprop(learning_rate=alpha))
        self.critic.compile(optimizer=keras.optimizers.RMSprop(learning_rate=beta))
        self.target_actor.compile(optimizer=keras.optimizers.RMSprop(learning_rate=alpha))
        self.target_critic.compile(optimizer=keras.optimizers.RMSprop(learning_rate=beta))
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)
        
    def save_weights_model(self):
        print('..... saving models .....')
        self.actor.save_weights(self.actor.checkpoint_file)
        self.critic.save_weights(self.critic.checkpoint_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.target_critic.save_weights(self.target_critic.checkpoint_file)
        
    def load_models(self):
        print('..... loading models .....')
        self.actor.build((1, 100))
        self.actor.load_weights(self.actor.checkpoint_file)
        # self.critic.load_weights(self.critic.checkpoint_file)
        # self.target_actor.load_weights(self.target_actor.checkpoint_file)
        # self.target_critic.load_weights(self.target_critic.checkpoint_file)
    
    def index_1D_to_2D(self, index):
        y = index % cols
        x = int(index / cols)
        return x,y
       
    def choose_action(self, state, evaluate=False):
        state = tf.convert_to_tensor([state], dtype = tf.float32)
        actions = self.actor(state)
        #if not when evaluating
        if not evaluate: 
            #add noise to action to critic
            actions += tf.random.normal(shape=[self.n_actions], mean=0.0, stddev=self.noise)
        return tf.math.argmax(actions[0]), actions[0]