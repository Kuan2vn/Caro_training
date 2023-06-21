import torch
import random
import numpy as np
from collections import deque
from gomoku import *
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

rows = 5
cols = 5
total_state = rows * cols

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        
        self.model = Linear_QNet(total_state, 256, total_state)
        

        # # LOAD MODEL PARAMETER FROM CHECKPOINT
        # self.model.load_state_dict(torch.load('over_time_model.pth'))

        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):
        state = game.Board()
        state = state.reshape(-1)
        return state

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)        

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 600 - self.n_games
        final_move = (0,0)
        if random.randint(0, 800) < self.epsilon:
            x = random.randint(0, rows-1)
            y = random.randint(0, cols-1)
            final_move = (x,y)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            print(state0.size)
            prediction = self.model(state0)
            
            i = torch.argmax(prediction).item() # action in 1D
            x, y = self.index_1D_to_2D(i)       # convert to 2D
            final_move = (x,y)

        return final_move
    
    def index_2D_to_1D(self, act):
      x, y = act
      return y + cols*x
      
    def index_1D_to_2D(self, index):
      y = index % cols
      x = int(index / cols)

      return x,y

def train():
    # plot_scores = []
    # plot_mean_scores = []
    # total_score = 0
    
    # plot_reward = []
    total_reward = 0

    move_taken = 0
    plot_move_taken = []
    plot_average_move_taken = []
    total_move_taken = 0


    game = Gomoku(rows=5, cols=5, n_to_win=3)

    agent = Agent()
    # agent2 = Agent()

    while True:
        state_old = agent.get_state(game)
        # print(state_old)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        # print(final_move)
        reward, done = game.play_step(final_move)
        state_new = agent.get_state(game)

        # # print Board
        # print('\n \nLast board:')
        # print(game.Board())

        total_reward += reward

        move_taken += 1

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        

        if done:

            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            # if move_taken < record:
            #      record = move_taken
            #      agent.model.save()

            if agent.n_games % 40 == 0:
                  agent.model.save2()

            # print('Game', agent1.n_games)

            # plot_scores.append(score)
            # total_score += score
            # mean_score = total_score / agent.n_games
            # plot_mean_scores.append(mean_score)
            # plot(plot_scores, plot_mean_scores)

            # plot_reward.append(total_reward)
            avg_reward = total_reward / agent.n_games
            plot_move_taken.append(move_taken)
            total_move_taken += move_taken
            mean_move_taken = total_move_taken / agent.n_games
            plot_average_move_taken.append(mean_move_taken)
            
            plot(plot_move_taken, plot_average_move_taken) #plot_move_taken2, plot_average_move_taken2)

            print('Average Reward Agent1: ',avg_reward)

            move_taken = 0



if __name__ == '__main__':
    train()