import pygame
import numpy as np
import time
from collections import deque
from snake_env import SnakeEnvironment

pygame.init()
font = pygame.font.SysFont('', 30)
apples_font = pygame.font.SysFont('', 30)
danger_text = font.render('Direction of danger', True, (255,255,255))
food_text = font.render('Direction(s) of food', True, (255,255,255))
display = pygame.display.set_mode((900,600))

def update_screen(board, obs, n_apples, model_type='mlp'):
    '''Function to display board and observation states through pygame'''
    display.fill((0))
    # display board
    for i,j in np.ndindex(board.shape):
        if board[i,j] == 0:
            pygame.draw.rect(display, (90,90,90), pygame.Rect(12*j, 12*i, 11, 11))
        elif board[i,j] == 1:
            pygame.draw.rect(display, (0,0,200), pygame.Rect(12*j, 12*i, 11, 11))
        else:
            pygame.draw.rect(display, (200,0,0), pygame.Rect(12*j, 12*i, 11, 11))
    if model_type == 'mlp':
        # display observations
        for i,active in enumerate(obs):
            horizontal_spacing = 0
            vertical_spacing = 0
            # left or right
            if i == 0 or i == 3:
                vertical_spacing = 52
                # left
                if i == 0:
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
            # bottom and top
            elif i == 6 or i == 7:
                horizontal_spacing = 52
                # bottom
                if  i == 6:
                    vertical_spacing = 52 * 2
                if active == 1:
                    pygame.draw.rect(display, (255,0,0), pygame.Rect(675+horizontal_spacing, 100+vertical_spacing, 50, 50))
                else:
                    pygame.draw.rect(display, (255,255,255), pygame.Rect(675+horizontal_spacing, 100+vertical_spacing, 50, 50))
            # food direction portion of the observation
            # right and left
            elif i == 8 or i == 9:
                vertical_spacing = 52
                # right
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
        
        apple_count_text = apples_font.render(f'Apples Eaten: {n_apples}', True, (255,255,255))
        display.blit(danger_text, (660, 50))
        display.blit(food_text, (660, 300))
        display.blit(apple_count_text, (680, 550))
    
    pygame.display.flip()

LOADED_MODEL = 'models/aggro boy/snakeDQN-50x50_ep9200_average361.9788043478261.pt'
MODEL_TYPE = 'mlp'

env = SnakeEnvironment(epsilon=0, loaded_model=LOADED_MODEL, model_type=MODEL_TYPE)
env.model.eval()

# import cv2
# usermove = None

for i in range(1000):
    apple_count = 0
    ep_reward = 0
    expected_reward = []
    curr_obs = env.reset()
    if MODEL_TYPE == 'conv':
        curr_obs = np.stack((curr_obs, curr_obs))
    
    while not env.done:
        
        events = pygame.event.get()
        # for event in events:
        #     if event.type == pygame.KEYDOWN:
        #         if event.key == pygame.K_LEFT:
        #             usermove = 0
        #         if event.key == pygame.K_RIGHT:
        #             usermove = 1
        #         if event.key == pygame.K_UP:
        #             usermove = 2
        #         if event.key == pygame.K_DOWN:
        #             usermove = 3
        
        q_val, action = env.move(curr_obs)
        # get next state update
        # oldmove = usermove
        new_obs, reward, _ = env.step(action)
        ep_reward += reward
        if q_val is not None:
            expected_reward.append(q_val)
        # for conv model
        if MODEL_TYPE == 'conv':
            new_obs = np.stack((new_obs, curr_obs[0]))
            
        # debugging
        # img = cv2.resize(new_obs, (300,300))
        # cv2.imshow('', img)
        # cv2.waitKey(100)
        
        # to display the agent playing the game
        update_screen(env.board, curr_obs, env.apple_count, model_type=MODEL_TYPE)
        
        curr_obs = new_obs 
        time.sleep(0.025)
    
    print(f'Apples: {env.apple_count}')
    print(f'Expected Reward: {np.array(expected_reward).mean()}')
    print(f'Actual Reward: {ep_reward}')