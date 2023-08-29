import numpy as np
from snake_env import SnakeEnvironment
from collections import deque
from tqdm import tqdm
import torch
import wandb

EPISODES = 100000

wandb.init(project="Snake DQN", entity="steez")
wandb.config = {
  "learning_rate": 1e-5,
  "epochs": EPISODES,
  "batch_size": 64
}

BOARD = np.zeros((50,50))
INITIAL_ESPILON = 0.99
MODEL_NAME = f'snakeDQN-{len(BOARD[0])}x{len(BOARD[1])}'
LOADED_MODEL = 'models/convdqn/snakeDQN-30x30_ep11100_average-0.990927927927928.pt'
MODEL_TYPE = 'mlp full'

env = SnakeEnvironment(epsilon=INITIAL_ESPILON, loaded_model=None, model_type=MODEL_TYPE)
print(sum(p.numel() for p in env.model.parameters()))


mavg_reward = deque(maxlen=25)
avg_reward = []
avg_q_val = []
n_gradient_updates = 0
n_steps = 0

wandb.watch(env.model, env.model.loss_fn, log='all', log_freq=10)

for i in tqdm(range(1,EPISODES+1)):
    episode_reward = 0
    curr_obs = env.reset()
    # obs variation for different model tests
    if MODEL_TYPE == 'conv':
        curr_obs = np.stack((curr_obs, curr_obs))

    while not env.done:
        q_value, action = env.move(curr_obs)
        # for tracking q values of actions
        if q_value is not None:
            avg_q_val.append(q_value)
            
        # get next state update
        new_obs, reward, got_apple = env.step(action)
        
        if MODEL_TYPE == 'conv':
            new_obs = np.stack((new_obs, curr_obs[0]))
            
            
        transition = (curr_obs, action, reward, new_obs, env.done)
                
        env.update_replay_memory(transition)
        env.train()
        # track number of gradient updates for loss tracking
        if env.is_training is True:
            n_gradient_updates += 1
        
        # decay epsilon 
        if n_steps % 400 == 0:
            env.decay_epsilon()
        
        curr_obs = new_obs
        
        n_steps += 1
        episode_reward += reward
        
    
    mavg_reward.append(episode_reward)
    avg_reward.append(episode_reward)
    
    wandb.log({'loss': (env.running_loss / n_gradient_updates) if env.is_training is True else 0,
               'apples eaten': env.apple_count,
               'reward': episode_reward,
               'average reward (last 25)': np.array(mavg_reward).mean(),
               'average q-value per step': np.array(avg_q_val).mean(),
               'epsilon': env.epsilon,})
    
    print()
    print(f'Reward for this episode: {episode_reward}')
    print(f'Apples: {env.apple_count}')
    print(f'Epsilon: {env.epsilon}')
    
    env.is_training = False
    
    if i % 100 == 0:
        if MODEL_TYPE == 'conv':
            torch.save(env.model.state_dict(), f'models/convdqn/{MODEL_NAME}_ep{i}_average{np.array(avg_reward).mean()}.pt')
        elif MODEL_TYPE == 'mlp':
            torch.save(env.model.state_dict(), f'models/aggro boy/{MODEL_NAME}_ep{i}_average{np.array(avg_reward).mean()}.pt')
        else:
            torch.save(env.model.state_dict(), f'models/mlp full board state/{MODEL_NAME}_ep{i}_average{np.array(avg_reward).mean()}.pt')
        

