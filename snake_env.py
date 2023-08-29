import numpy as np
import torch
from snake_model import SnakeDQN, SnakeConvDQN
from collections import deque
import random

# parameters
BOARD_SIZE = (50,50)
MAX_MEMORY = 100000
MIN_MEMORY = MAX_MEMORY / 10
BATCH_SIZE = 64
DISCOUNT = .9999
UPDATE_FREQ = 7500 # n steps before target network update
MIN_EPSILON = 0.01
DECAY_FACTOR = 0.99

class SnakeEnvironment():
    def __init__(self, epsilon=1, loaded_model=None, model_type='mlp'):
        # array that represents the board state
        self.board = np.zeros(BOARD_SIZE)
        # tracks snake's body
        self.snake_body = deque()
        # these are the x,y coordinates of the snake's head
        self.head_y, self.head_x = (0,0)
        self.apple_count = 0
        self.food_pos = None
        # using this to add diminishing returns on moving closer to apple
        self.steps_since_apple = 0
        # terminal state flag
        self.done = False
        self.model_type = model_type
        self.model = self.build_model(self.model_type)
        if loaded_model is not None:
            self.model.load_state_dict(torch.load(loaded_model))
        self.target_model = self.build_model(self.model_type)
        self.target_model.load_state_dict(self.model.state_dict())
        # moving models to GPU
        self.model.cuda()
        self.target_model.cuda()
        # counter to track how often the target model will be update to the weights of the training model
        self.target_update_counter = 0
        # internal tracking of current loss and epsilon as well as whether the model has started training
        self.is_training = False
        self.running_loss = 0
        self.epsilon = epsilon
        # experience replay memory 
        self.memory = deque(maxlen=MAX_MEMORY)
        self.important_memories = None
        
    def reset_board(self):
        self.board = np.zeros(self.board.shape)
        
    def is_game_over(self, next_x, next_y):
        '''
            Function to check terminal state of the episode
        '''
        if (next_y,next_x) in self.snake_body or next_x < 0 or next_x > self.board.shape[1]-1 or next_y < 0 or next_y > self.board.shape[0]-1:
            self.done = True
            return self.done
        else:
            return self.done
    
    def is_collison(self, next_x, next_y):
        '''
            Function that checks collision of snake with edges of the board and with the snake's own body
        '''
        if (next_y,next_x) in self.snake_body or next_x < 0 or next_x > self.board.shape[1]-1 or next_y < 0 or next_y > self.board.shape[0]-1:
            return True
        else:
            return False
    
    def spawn_food_on_board(self):
        '''
            Spawns food randomly onto board at each call
        '''
        rd = (np.random.randint(len(self.board)-1), np.random.randint(len(self.board[0])-1))
        while rd in self.snake_body:
            rd = (np.random.randint(len(self.board)-1), np.random.randint(len(self.board[0])-1))
        # spawn food randomly
        for i, j in np.ndindex(self.board.shape):
            if (i,j) == rd:
                self.board[i,j] = 2
                self.food_pos = (i,j) # y,x
                break
    
    def move(self, state):
        '''
            Function that implements epsilon-greedy action policy
        '''
        if np.random.random() < self.epsilon:
            choice = np.random.randint(0,4)
            q_value = None
        else:
            self.model.eval()
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32).cuda()
                prediction = self.model(state.view(-1, *state.shape))
                q_value, choice = torch.max(prediction, dim=1)
            self.model.train()
            q_value = q_value.cpu().numpy()
            choice = choice.cpu().numpy()
            
        return q_value, choice
        
    
    def step(self, action):
        '''
            Function to update environment state step-by-step
            Returns rewards and observations
        '''
        # baseline reward per timestep for doing nothing prodcutive
        self.steps_since_apple += 1
        # left
        if action == 0:
            self.head_x -= 1
        # right
        elif action == 1:
            self.head_x += 1
        # up
        elif action == 2:
            self.head_y -= 1
        # down
        else:
            self.head_y += 1
            
        # check game over
        self.is_game_over(self.head_x, self.head_y)
        # if no apple for too long -- matters less if snake gets big
        # if self.steps_since_apple > (30 * len(self.snake_body)):
        #     self.done = True
        
        if (self.head_y,self.head_x) == self.food_pos:
            self.snake_body.append((self.head_y, self.head_x))
            
            for part in self.snake_body:
                self.board[part] = 1
            
            self.spawn_food_on_board()
            
            if self.model_type == 'mlp':
                obs = self.get_observation()
            elif self.model_type == 'conv':
                obs = self.get_full_obs()
            elif self.model_type == 'mlp full':
                obs = self.get_full_obs()
            
            reward = len(self.snake_body) - 3 # - 3 for starting length
            self.apple_count += 1 
            self.steps_since_apple = 0
            # true for apple
            return obs, reward
        
        if not self.done:
            # game not over
            self.board[self.head_y,self.head_x] = 1
            self.snake_body.append((self.head_y,self.head_x))
            self.board[self.snake_body.popleft()] = 0
            reward = 0
        else:
            # game over
            if self.head_x < self.board.shape[1] and self.head_y < self.board.shape[0] and self.head_x > 0 and self.head_y > 0:
                self.board[self.head_y,self.head_x] = 1
                self.snake_body.append((self.head_y,self.head_x))
                self.board[self.snake_body.popleft()] = 0
                
            reward = -1 
        
        if self.model_type == 'mlp':
            obs = self.get_observation()
        elif self.model_type == 'conv':
            obs = self.get_full_obs()
        elif self.model_type == 'mlp full':
            obs = self.get_full_obs()
            
        return obs, reward
    
    def get_observation(self):
        '''
            Get observation to feed into the DQN.
            Returns binary array of variables representing directions of danger and food
        '''
        observation = [
            # where is the danger
            int(self.is_collison(self.head_x+1, self.head_y)),    # danger right
            int(self.is_collison(self.head_x+1, self.head_y-1)),  # danger top right
            int(self.is_collison(self.head_x+1, self.head_y+1)),  # danger bottom right
            int(self.is_collison(self.head_x-1, self.head_y)),    # danger left
            int(self.is_collison(self.head_x-1, self.head_y-1)),  # danger top left
            int(self.is_collison(self.head_x-1, self.head_y+1)),  # danger bottom left
            int(self.is_collison(self.head_x, self.head_y+1)),    # danger down
            int(self.is_collison(self.head_x, self.head_y-1)),    # danger up
            # where is the food
            int(self.head_x < self.food_pos[1]), # food right
            int(self.head_x > self.food_pos[1]), # food left
            int(self.head_y < self.food_pos[0]), # food down
            int(self.head_y > self.food_pos[0])  # food up
        ]
        return np.array(observation)
    
    def get_full_obs(self):
        '''
            Gets full board state obs
        '''
        if self.model_type == 'conv':
            frame_array = np.zeros(self.board.shape)
            frame_array[self.food_pos] = 255
            body_pos = [yx for yx in self.snake_body]
            
            for pos in body_pos:
                frame_array[pos] = 127
            # changing pixel value of head
            frame_array[self.snake_body[-1]] = 191
            
            obs = frame_array / 255
            
        if self.model_type == 'mlp full':
            danger = []
            for i in range(-10,10):
                for j in range(-10,10):
                    danger.append(int(self.is_collison(self.head_x+j, self.head_y+i)))
                    
            food_dirs = [
                int(self.head_x < self.food_pos[1]), # food right
                int(self.head_x > self.food_pos[1]), # food left
                int(self.head_y < self.food_pos[0]), # food down
                int(self.head_y > self.food_pos[0])  # food up
            ]
            
            for dir in food_dirs:
                danger.append(dir)
            
            obs = np.array(danger)
        
        return obs
    
    def reset(self):
        '''
            Resets environment
        '''
        self.reset_board()
        # self.food_pos = (int(BOARD_SIZE[0]/2), int(BOARD_SIZE[1]/3))
        # self.board[self.food_pos] = 2
        spawny, spawnx = (int(BOARD_SIZE[0]/2), 2 * int(BOARD_SIZE[1]/3))
        self.snake_body = deque([(spawny, spawnx), (spawny, spawnx+1), (spawny, spawnx+2)])
        # putting snake onto board
        for part in self.snake_body:
            self.board[part] = 1
        self.spawn_food_on_board()
        self.head_y, self.head_x = self.snake_body[-1]
        self.done = False
        self.apple_count = 0
        self.steps_since_apple = 0
        
        if self.model_type == 'mlp':
            obs = self.get_observation()
        elif self.model_type == 'conv':
            obs = self.get_full_obs()
        elif self.model_type == 'mlp full':
            obs = self.get_full_obs()
        
        return obs
    
    def train(self):
        '''
            Training-step function
        '''
        if len(self.memory) < MIN_MEMORY:
            return 0
        # training flag
        self.is_training = True
        # sampling a batch from the replay buffer
        sample = random.sample(self.memory, BATCH_SIZE)
        
        self.model.eval()
        self.target_model.eval()
        with torch.no_grad():
            # fishing out current states from the batch, and passing them through the model to get current estimated q-values
            current_states = torch.tensor(np.array([transition[0] for transition in sample]),
                                          dtype=torch.float32).cuda()
            current_qs = self.model(current_states).cpu().detach().numpy()
            # same thing but for future q-values predicted from next states
            new_states = torch.tensor(np.array([transition[3] for transition in sample]),
                                      dtype=torch.float32).cuda()
            future_qs = self.target_model(new_states).cpu().detach().numpy()
        
        self.model.train()
        
        x = []
        y = []
        
        for idx, (current_state, action, reward, new_state, done) in enumerate(sample):
            # computing bellman
            if not done:
                max_future_q = np.max(future_qs[idx])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward
            
            current_q = current_qs[idx]
            current_q[action] = new_q
            
            x.append(current_state)
            y.append(current_q)
            
        # fitting a single batch of data
        self.running_loss += self.model.fit(torch.tensor(np.array(x), dtype=torch.float32),
                                            torch.tensor(np.array(y), dtype=torch.float32))
        
        # episode counter for updating target model
        self.target_update_counter += 1
        if self.target_update_counter > UPDATE_FREQ:
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_update_counter = 0
    
    def update_replay_memory(self, transition):
        '''
            This just adds a game step to the replay buffer
        '''
        self.memory.append(transition)
    
    def decay_epsilon(self):
        # epsilon decay
        if self.epsilon > MIN_EPSILON:
            self.epsilon *= DECAY_FACTOR
            self.epsilon = max(MIN_EPSILON, self.epsilon)
    
    def build_model(self, type='mlp'):
        if type == 'mlp':
            model = SnakeDQN()
        elif type == 'conv':
            model = SnakeConvDQN()
        return model
    
