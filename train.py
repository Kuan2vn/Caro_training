from helper import plot_as
from gomoku import Gomoku
from agent import Agent
import numpy as np

if __name__ == '__main__':
    env = Gomoku(rows=5, cols=5, n_to_win=3)
    N = 20            # move taken before learning
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    agent = Agent(n_actions=env.matrix, batch_size=batch_size, 
                    alpha=alpha, n_epochs=n_epochs, 
                    input_dims=env.matrix)
    n_games = 300


    learn_iters = 0
    
    n_steps = 0

    move_taken = 0
    plot_move_taken = []
    
    total_move_taken = 0

    for i in range(n_games):
        env.reset()
        observation = env.get_state()
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            reward, done = env.take_action(action)
            observation_ = env.get_state()
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_

            move_taken += 1


        plot_move_taken.append(move_taken)
        total_move_taken += move_taken
        mean_move_taken = total_move_taken / n_games
        
        print('average move taken/ game: ', mean_move_taken)
        plot_as(plot_move_taken)
        move_taken = 0
        
        if i % 40 == 0:
            agent.save_models()