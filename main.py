
from agent import *
from gomoku import *
import time

import numpy as np

from gomoku import Gomoku

from agent import Agent
from predict import predict




pygame.init()
env = Gomoku(rows=10, cols=10, n_to_win=5)
N = 16           # move taken before learning
batch_size = 32
n_epochs = 16
count = 1

agent1 = Agent(n_actions=env.matrix, batch_size=batch_size, 
                n_epochs=n_epochs, 
                input_dims=env.matrix, chkpt_dir='Tu/agent1/ppo')
agent2 = Agent(n_actions=env.matrix, batch_size=batch_size, 
                n_epochs=n_epochs, 
                input_dims=env.matrix, chkpt_dir='Tu/agent2/ppo')

# agent1.save_models
agent1.load_models()
agent2.load_models()
agent = [agent1, agent2]

pygame.time.Clock().tick(60)
# self.draw_board()

env.player_names["b"] = "Black"
env.player_names["w"] = "White"

while not env.board.just_won() and not env.board.is_draw():
    time.sleep(0.6)
    pygame.display.update()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
    # env.action((0,0))
    # env.action((0,1))
    
    if count == 1:
        # state = env.get_state()
        # x,y = predict(agent1, state)

        # done = env.action((x,y))
        # if done:
        #     count = (count + 1) % 2

        state = env.get_state()
        att = agent1.predict(state)
        x, y = att
        done = env.action((x,y))
        print(done)
        if done:
            count = (count + 1) % 2

    else: 
    # print(att)
        state = env.get_state()
        att = agent2.predict(state)
        x, y = att
        done = env.action((x,y))
        # print(done)
        if done:
            count = (count + 1) % 2
    pygame.display.update()
   

#     count = (count + 1) % 2
print(env.Board())

env.play()
# env.show_outcome()
# pygame.display.update()
# env.exit_on_click()
    






