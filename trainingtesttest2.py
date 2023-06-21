from agent import *
from gomoku import *
import time

if __name__ == '__main__':
    env = Gomoku(rows=10, cols=10, n_to_win=5)
    N = 8           # move taken before learning
    batch_size = 32
    n_epochs = 16

    count = 1

    agent1 = Agent(n_actions=env.matrix, batch_size=batch_size,
                    n_epochs=n_epochs,
                    input_dims=env.matrix, chkpt_dir='agent1/ppo')
    agent2 = Agent(n_actions=env.matrix, batch_size=batch_size,
                    n_epochs=n_epochs,
                    input_dims=env.matrix, chkpt_dir='agent2/ppo')
    agent1.load_models()
    agent2.load_models()
    n_games = 1000

    # figure_file = 'plots/cartpole.png'

    # best_score = env.reward_range[0]
    score_history1 = []
    score_history2 = []

    cheat_move1 = []
    cheat_move2 = []

    learn_iters1 = 0
    learn_iters2 = 0

    avg_score1 = 0
    avg_score2 = 0

    agent1_wins = 0
    agent2_wins = 0

    best_score1 = 0
    best_score2 = 0

    n_steps1 = 0
    n_steps2 = 0
    temp_reward = 0

    move_taken = 0
    plot_move_taken = []
    # plot_average_move_taken = []
    # total_move_taken = 0

    for i in range(n_games):
        env.reset()
        # observation = env.get_state()
        done = False
        done_action = False
        done_action2 = False
        score1 = 0
        score2 = 0
        print('Game number ',i)
        while not done:

            done_action = False
            done_action2 = False
            temp = 0

            observation2 = env.get_state()
            # player 2
            while not done_action2:
              action2, probs2, val2 = agent2.choose_action(observation2)
              reward2, done, done_action2 = env.take_action(action2)
              # print('agent2 took', action2)
              score2 += reward2
              if score2 >= 500:
                cheat_move2.append(action2)

              temp += 1
              if temp <= 30:
                agent2.remember(observation2, action2, probs2, val2, reward2, done)
              if done_action2 == True:
                break
              else:
                print('agent2 FALSE MOVE')
            
              if temp > 30:
                action2, _,_ = agent1.choose_action(observation2)
                _, done, done_action2 = env.take_action(action2)
                if done_action2 == True:
                    print('agent1 save the day')
                    break                    
                else:
                    print('agent1 failed')
              if temp > 50:
                action2 = np.random.choice(env.matrix)
                _, done, done_action2 = env.take_action(action2)
                if done_action2 == True:
                    print('random saved the day')
                    break                    
                else:
                    print('random failed')
                        
                           
            n_steps2 += 1
            if n_steps2 % N == 0:
                agent2.learn()
                learn_iters2 += 1
                print('agent2 learn')
            
            time.sleep(0.6)

            if done == True:
                break

            observation = env.get_state()
            
            # player 1
            while not done_action:
              action, prob, val = agent1.choose_action(observation)
              reward1, done, done_action = env.take_action(action)
              # print('agent1 took',action)
              score1 += reward1
              if reward1 >= 100:
                  agent1_wins += 1
              if score1 > 500:
                cheat_move1.append(action)


              temp += 1
              if temp <= 30:
                agent1.remember(observation, action, prob, val, reward1, done)
              if done_action == True:
                break
              else:
                print('agent1 FALSE MOVE')

              if temp > 30:
                action, _,_ = agent2.choose_action(observation)
                _, done, done_action = env.take_action(action)
                if done_action == True:
                  print('agent2 saved the day')
                  break
                else:
                  print('agent2 save failed')
              if temp > 50:
                action = np.random.choice(env.matrix)
                _, done, done_action = env.take_action(action)
                if done_action == True:
                  print('random saved the day')
                  break
                else:
                  print('random save failed')



            
            # print(env.index_1D_to_2D(action))

            n_steps1 += 1

            move_taken += 1

            if n_steps1 % N == 0:
                agent1.learn()
                learn_iters1 += 1
                print('agent1 learn')
            
            time.sleep(0.6)

            

            temp = 0
            

            # env.show_outcome()
            pygame.display.update()
            # env.exit_on_click()


            

        # plot_move_taken.append(move_taken)
        # total_move_taken += move_taken
        # mean_move_taken = total_move_taken / n_games



        # move_taken = 0
        # print('number games ', i)
        # if i % 5 == 0 and i != 0:
            #   agent1.save_models()
            #   agent2.save_models()
        #       !cp /content/agent1/ppo/actor/actor.pth /content/drive/MyDrive/gomoku_model/agent1
        #       !cp /content/agent1/ppo/critic/critic.pth /content/drive/MyDrive/gomoku_model/agent1
        #       !cp /content/agent2/ppo/actor/actor.pth /content/drive/MyDrive/gomoku_model/agent2
        #       !cp /content/agent2/ppo/critic/critic.pth /content/drive/MyDrive/gomoku_model/agent2
        # if score1 >= best_score1:
        #       !cp /content/agent1/ppo/actor/actor.pth /content/drive/MyDrive/gomoku_model/agent1_best
        #       !cp /content/agent1/ppo/critic/critic.pth /content/drive/MyDrive/gomoku_model/agent1_best
        #       best_score1 = score1
        # if score2 >= best_score2:
        #       !cp /content/agent2/ppo/actor/actor.pth /content/drive/MyDrive/gomoku_model/agent2_best
        #       !cp /content/agent2/ppo/critic/critic.pth /content/drive/MyDrive/gomoku_model/agent2_best
        #       best_score2 = score2

        # score_history1.append(score1)
        # score_history2.append(score2)

        # avg_score1 = np.mean(score_history1[-100:])
        # avg_score2 = np.mean(score_history2[-100:])

        # plot_as(plot_move_taken, score_history1, score_history2)

        # print('average move taken/ game: ', mean_move_taken)
        # print('agent1 average score: ', avg_score1)
        # print('agent2 average score: ', avg_score2)
        # print('--------------------')
        # print('total game played: ', i+1)
        # print('agent1 total wins: ', agent1_wins)
        # print('agent2 total wins: ', agent2_wins)


        # if avg_score > best_score:
        #     best_score = avg_score
        #     agent.save_models()

        # print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                # 'time_steps', n_steps, 'learning_steps', learn_iters)
    # x = [i+1 for i in range(len(score_history))]
    # plot_learning_curve(x, score_history, figure_file)