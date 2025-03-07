## Required dependencies
<table>
  <tr>
    <td>
    - Python version 3.9 (<a href="https://www.python.org/">python</a>)
    </td>
  </tr>
    <td>
      - Pygame lastest version (<a href="https://www.pygame.org/">pygame</a>)
    </td>
  <tr>
    <td>
      - Pytorch lastest version (<a href="https://pytorch.org/">PyTorch</a>)
    </td>
  </tr>
</table>

## Build a model with PyTorch

To unify, we decided to use **PyTorch** as a common framework to build deep learning neural networks. There are different ways to install **PyTorch** depending on your operating system and your preferences. You can visit the official website of **PyTorch** at https://pytorch.org/ and select your preferences to run the install command. Then you can start build a deep learning model with **PyTorch** framework, in here I’m only build a simple linear network with 3 layers, but you can have more layers with larger units at each layer.

## The environment/ Model input

The Gomoku game usually uses a 10x10 board to play and the first player who can place 5 consecutive pieces horizontally, vertically or diagonally will be the winner. So the state of the environment will be a 2D numpy array with a dimension of 10x10 , with the number 0 used to indicate that the corresponding cell is empty, 1 indicates that the cell has been marked by a white piece and -1 indicates that the cell has been marked by a black piece. To use this as a input to the model, the state will flatten to a 1D numpy array with 1x100 dimension, you can get the 1D numpy array with the function *env.get_state()*. Or if you wish to feed the model with 2D array, you can delete the reshape function in the *get_state* function in *gomoku.py* file.

## Take action/ Model output (For DQN)

The final output of the model is the Q value of 100 cells, and the model will choose the index of the cell with the highest Q value and use that as the action to take. But the output of the model is a variable that is the index in a 1D matrix, so we will have to use the function *index_1D_to_2D* to transfer that index into 2 variables that determine which column and row is that in the corresponding 2D matrix. Then the model can take action with the function *action(rows, cols)* or take action and learn the reward for that action with the function *take_action(rows, cols)* which will return the reward and whether that action will end the game or not. You can also tweak this function in *gomoku.py* to change the reward that is more suitable for your model.

## Save the model

When the model completed training, we can save the model with the function *agent.save_models()*, which will create a folder named fast_model and save the current model state as a .pth file. Or you can save the model manually with the function *torch.save(self.state_dict(), file_name)*.

## Load Model

To load the model saved state with a .pth file, you must first create a model with the same amount of layers and units per layer, then you can load the last model state with the function self.model.load_state_dict(torch.load(‘.pth’)). 

## Publication

**Project published at VNICT 2023.**

**[Deep reinforcement learning for Playing Caro Game without human feedback](DRL_Caro.pdf)**

## Summary
Finally, to test the training code, you can run the *train.py* 
<br>Or, you can find training example in colab *Actor_Critic_Example.ipynb.* and my old DQN model in Example folder.
