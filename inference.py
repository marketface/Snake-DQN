import pygame
import numpy as np
import time
from collections import deque
from snake_env import SnakeEnvironment

pygame.init()
pygame.font.init()
font = pygame.font.SysFont('Calibri', 30)
danger_text = font.render('Direction of danger', True, (255,255,255))
food_text = font.render('Direction(s) of food', True, (255,255,255))
display = pygame.display.set_mode((900,600))
display.blit(danger_text, (630, 50))
display.blit(food_text, (630, 300))

def update_screen(board, obs):
    '''Function to display board and observation states through pygame'''
    # display board
    for i,j in np.ndindex(board.shape):
        if board[i,j] == 0:
            pygame.draw.rect(display, (90,90,90), pygame.Rect(12*i, 12*j, 11, 11))
        elif board[i,j] == 1:
            pygame.draw.rect(display, (0,0,200), pygame.Rect(12*i, 12*j, 11, 11))
        else:
            pygame.draw.rect(display, (200,0,0), pygame.Rect(12*i, 12*j, 11, 11))
    # display observations in real time
    # the code is really ugly because i thought of implementing this after I trained the model.
    # because of this, the observations are disordered and require some ridiculous conditional logic to make work (at least for me lol)
    for i,active in enumerate(obs):
        horizontal_spacing = 0
        vertical_spacing = 0
        # left or right
        if i == 0 or i == 3:
            vertical_spacing = 52
            # left
            if i == 3:
                horizontal_spacing = 52 * 2
                
            if active == 1:
                pygame.draw.rect(display, (255,0,0), pygame.Rect(675+horizontal_spacing, 100+vertical_spacing, 50, 50))
            else:
                pygame.draw.rect(display, (255,255,255), pygame.Rect(675+horizontal_spacing, 100+vertical_spacing, 50, 50))
        # top right and bottom right
        elif i == 1 or i == 2:
            horizontal_spacing = 52*2
            # bottom right
            if i == 2:
                vertical_spacing = 52 * 2
                
            if active == 1:
                pygame.draw.rect(display, (255,0,0), pygame.Rect(675+horizontal_spacing, 100+vertical_spacing, 50, 50))
            else:
                pygame.draw.rect(display, (255,255,255), pygame.Rect(675+horizontal_spacing, 100+vertical_spacing, 50, 50))
        elif i == 4 or i == 5:
            horizontal_spacing = 0
            # bottom left
            if i == 5:
                vertical_spacing = 52*2
            else:
                vertical_spacing = 0
                
            if active == 1:
                pygame.draw.rect(display, (255,0,0), pygame.Rect(675+horizontal_spacing, 100+vertical_spacing, 50, 50))
            else:
                pygame.draw.rect(display, (255,255,255), pygame.Rect(675+horizontal_spacing, 100+vertical_spacing, 50, 50))
        # top and bottom
        elif i == 6 or i == 7:
            horizontal_spacing = 52
            # bottom
            if  i == 7:
                vertical_spacing = 52 * 2
            if active == 1:
                pygame.draw.rect(display, (255,0,0), pygame.Rect(675+horizontal_spacing, 100+vertical_spacing, 50, 50))
            else:
                pygame.draw.rect(display, (255,255,255), pygame.Rect(675+horizontal_spacing, 100+vertical_spacing, 50, 50))
        # food direction portion of the observation
        # right and left
        elif i == 8 or i == 9:
            vertical_spacing = 52
            if i == 8:
                horizontal_spacing = 52 * 2
            if active == 1:
                pygame.draw.rect(display, (0,255,0), pygame.Rect(675+horizontal_spacing, 350+vertical_spacing, 50, 50))
            else:
                pygame.draw.rect(display, (255,255,255), pygame.Rect(675+horizontal_spacing, 350+vertical_spacing, 50, 50))
        # down and up
        elif i > 9:
            horizontal_spacing = 52
            if i == 10:
                vertical_spacing = 52 * 2
            if active == 1:
                pygame.draw.rect(display, (0,255,0), pygame.Rect(675+horizontal_spacing, 350+vertical_spacing, 50, 50))
            else:
                pygame.draw.rect(display, (255,255,255), pygame.Rect(675+horizontal_spacing, 350+vertical_spacing, 50, 50))
        
    pygame.display.flip()

BOARD = np.zeros((50,50))
LOADED_MODEL = 'models/10000 epoch run (50x50)/model-50x50-max560-average115.0-ep4800.pt'

board = np.zeros((len(BOARD), len(BOARD[0])))
snake = SnakeEnvironment(deque([(np.random.randint(0,50), np.random.randint(0,50))]), board=board, epsilon=0, loaded_model=LOADED_MODEL)

for i in range(1000):
    apple_count = 0
    current_obs = snake.reset()
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
        
        # if the snake takes more than 300 steps without getting food the episode is reset to speed up training
        timesteps += 1
        if timesteps > 300:
            break
        
        # to display the agent playing the game
        update_screen(board, current_obs)
            
        current_obs = new_obs
        
        time.sleep(0.025)
    
    print(f'Apples: {apple_count}')