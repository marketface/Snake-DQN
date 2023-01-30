import numpy as np
import pygame
from snake_env import SnakeEnvironment
from collections import deque
from tqdm import tqdm
import torch
import wandb

# TODO: reformat code to easily change between training/testing modes
# this includes changing epsilon as such and making model loading easier

wandb.init(project="Snake DQN", entity="steez")
wandb.config = {
  "learning_rate": 0.001,
  "epochs": 10000,
  "batch_size": 256
}


TRAINING = True
BOARD = np.zeros((50,50))
INITIAL_ESPILON = 0.01
MODEL_NAME = f'model-{len(BOARD)}x{len(BOARD[0])}'
LOADED_MODEL = 'models/10000 epoch run (50x50)/model-50x50-max560-average115.0-ep4800.pt'

pygame.init()
display = pygame.display.set_mode((600,600))
def update_screen(board):
    '''Simple function to display board array through pygame'''
    for i,j in np.ndindex(board.shape):
        if board[i,j] == 0:
            pygame.draw.rect(display, (90,90,90), pygame.Rect(12*i, 12*j, 11, 11))
        elif board[i,j] == 1:
            pygame.draw.rect(display, (0,0,200), pygame.Rect(12*i, 12*j, 11, 11))
        else:
            pygame.draw.rect(display, (200,0,0), pygame.Rect(12*i, 12*j, 11, 11))
    pygame.display.flip()

board = np.zeros((len(BOARD), len(BOARD[0])))
snake = SnakeEnvironment(deque([(np.random.randint(0,50), np.random.randint(0,50))]), board=board, epsilon=INITIAL_ESPILON, loaded_model=LOADED_MODEL)

episodes = 10000
average_score = []
apples_per_run = []
n_gradient_updates = 0

for i in tqdm(range(1,episodes+1)):
    apple_count = 0
    current_obs = snake.reset()
    episode_reward = 0
    timesteps = 0
    # to prevent division by 0 when logging loss while replay buffer is filling with data
    
    while not snake.done:
        action = snake.move(current_obs)
        # get next state update
        board, new_obs, reward, is_new_food = snake.step(action)
        # check for food on the board
        if is_new_food:
            timesteps = 0
            apple_count += 1
            snake.spawn_food_on_board()
            
        episode_reward += reward
        
        snake.update_replay_memory((current_obs, action, reward, new_obs, snake.done))
        snake.train(snake.done)
        # this is to track when the model starts training and count the number of gradient updates for the episode
        if snake.is_training is True:
            n_gradient_updates += 1
        
        # if the snake takes more than 300 steps without getting food the episode is reset to speed up training
        timesteps += 1
        if timesteps > 300:
            break
        
        # to display the agent playing the game
        # if i % 20 == 0:
        update_screen(board)
            
        current_obs = new_obs
        
    print()
    print(f'Reward for this episode: {episode_reward}')
    print(f'Apples: {apple_count}')
    print(f'Epsilon: {snake.epsilon}')
    
    apples_per_run.append(apple_count)    
    average_score.append(episode_reward)
    snake.is_training = False
    
    wandb.log({'loss': (snake.running_loss / n_gradient_updates) if snake.is_training is True else 0,
               'number of apples eaten': apple_count,
               'reward': episode_reward})
    
    if i % 100 == 0:
        average_reward = sum(average_score[-100:])/len(average_score[-100:])
        min_reward = min(average_score[-100:])
        max_reward = max(average_score[-100:])
        
        print()
        print(f'Average of 100 episodes: {average_reward} -- min: {min_reward} -- max: {max_reward}')
        print(f'Most apples in the last 100 runs: {max(apples_per_run[-100:])}')
        print(f'Epsilon: {snake.epsilon}')
        
        torch.save(snake.model.state_dict(), f'models/10000 epoch run (50x50)/{MODEL_NAME}-max{max_reward}-average{average_reward}-ep{i}.pt')