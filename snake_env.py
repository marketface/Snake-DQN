import numpy as np
import torch
from snake_model import SnakeDQN
from collections import deque
import random

# parameters
BOARD = np.zeros((50,50))
MAX_MEMORY = 100000
MIN_MEMORY = 1000
BATCH_SIZE = 250
DISCOUNT = .95
UPDATE_FREQ = 5

class SnakeEnvironment():
    '''
        SnakeEnvironment consists of all modules to run the game of snake and DQN training of the agent
    '''
    def __init__(self, body, board, epsilon=1, loaded_model=None):
        # tracks snake's body
        self.snake_body = body
        # these are the x,y coordinates of the snake's head
        self.snake_x, self.snake_y = body[-1]
        # array that represents the board state
        self.board = board
        # reward for the q-learning algorithm
        self.reward = 0
        # terminal state flag
        self.done = False
        
        self.model = self.build_model()
        if loaded_model is not None:
            self.model.load_state_dict(torch.load(loaded_model))
        self.target_model = self.build_model()
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
        
    
    def is_game_over(self, next_x, next_y):
        '''Function to check terminal state of the episode'''
        if (next_x,next_y) in self.snake_body or next_x < 0 or next_x > len(self.board)-1 or next_y < 0 or next_y > len(self.board[0])-1:
            self.done = True
            return self.done
        else:
            return self.done
    
    def is_collison(self, next_x, next_y):
        '''Function that checks collision of snake with edges of the board and with the snake's own body'''
        if (next_x,next_y) in self.snake_body or next_x < 0 or next_x > len(self.board)-1 or next_y < 0 or next_y > len(self.board[0])-1:
            return True
        else:
            return False
    
    def spawn_food_on_board(self):
        '''Spawns food randomly onto to board at each call'''
        rd = (np.random.randint(len(self.board)), np.random.randint(len(self.board[0])))
        # spawn food randomly
        for i, j in np.ndindex(self.board.shape):
            if (i,j) == rd:
                self.board[i,j] = 2
                self.food_pos = (i,j)
                break
    
    def move(self, state):
        '''Function that implements epsilon-greedy action policy'''
        if np.random.random() < self.epsilon:
            choice = np.random.randint(0,4)
        else:
            state = torch.tensor(state, dtype=torch.float32).cuda()
            prediction = self.model.forward(state)
            choice = torch.argmax(prediction).cpu()
        # epsilon decay which stops after epsilon reaches 0.01
        if self.epsilon > 0.01:
            self.epsilon *= 0.998
            self.epsilon = max(0.01, self.epsilon)
            
        return choice
        
    
    def step(self, action):
        '''Function to update environment state step-by-step'''
        is_new_food = False
        self.reward = 0
        # left
        if action == 0:
            self.snake_x -= 1
        # right
        elif action == 1:
            self.snake_x += 1
        # up
        elif action == 2:
            self.snake_y -= 1
        # down
        else:
            self.snake_y += 1
        
        self.is_game_over(self.snake_x, self.snake_y)
        
        obs = self.get_observation()
        
        if (self.snake_x,self.snake_y) == self.food_pos:
            self.snake_body.append((self.snake_x, self.snake_y))
            for coord in self.snake_body:
                self.board[coord] = 1
            is_new_food = True
            self.reward = 10
            return self.board, obs, self.reward, is_new_food
        
        if not self.done:
            self.board[self.snake_x,self.snake_y] = 1
            self.snake_body.append((self.snake_x,self.snake_y))
            self.board[self.snake_body.popleft()] = 0
        if self.done:
            self.reward = -10
        
        return self.board, obs, self.reward, is_new_food
    
    def get_observation(self):
        '''Get observation to feed into the DQN.
           These are binary variables in an array representing directions of danger and food'''
        observation = [
            # where is the danger
            int(self.is_collison(self.snake_x+1, self.snake_y)),    # danger right
            int(self.is_collison(self.snake_x+1, self.snake_y-1)),  # danger top right
            int(self.is_collison(self.snake_x+1, self.snake_y+1)),  # danger bottom right
            int(self.is_collison(self.snake_x-1, self.snake_y)),    # danger left
            int(self.is_collison(self.snake_x-1, self.snake_y-1)),  # danger top left
            int(self.is_collison(self.snake_x-1, self.snake_y+1)),  # danger bottom left
            int(self.is_collison(self.snake_x, self.snake_y+1)),    # danger down
            int(self.is_collison(self.snake_x, self.snake_y-1)),    # danger up
            # where is the food
            int(self.snake_x < self.food_pos[0]), # food right
            int(self.snake_x > self.food_pos[0]), # food left
            int(self.snake_y < self.food_pos[1]), # food down
            int(self.snake_y > self.food_pos[1])  # food up
        ]
        return np.array(observation)
    
    def reset(self):
        '''Resets environment'''
        self.board = np.zeros((len(BOARD), len(BOARD[0])))
        spawnx, spawny = (np.random.randint(len(BOARD)-2), np.random.randint(len(BOARD[0]-2)))
        self.snake_body = deque([(spawnx, spawny), (spawnx+1, spawny), (spawnx+2, spawny)])
        self.snake_x, self.snake_y = self.snake_body[-1]
        self.done = False
        self.spawn_food_on_board()
        
        # create first observation
        obs = self.get_observation()
        
        return obs
    
    def train(self, terminal_state):
        '''Training step function'''
        # first check that the experience replay buffer has enough data to start gradient descent updates on DQN
        if len(self.memory) < MIN_MEMORY:
            return 
        else:
            # training flag
            self.is_training = True
            # sampling a batch from the replay buffer
            sample = random.sample(self.memory, BATCH_SIZE)
            # fishing out current states from the batch, and passing them through the model to get current estimated q-values
            current_states = torch.tensor(np.array([transition[0] for transition in sample]), dtype=torch.float32, requires_grad=False).cuda()
            current_qs = self.model.forward(current_states).cpu().detach().numpy()
            # same thing but for future q-values predicted from next states
            new_states = torch.tensor(np.array([transition[3] for transition in sample]), dtype=torch.float32, requires_grad=False).cuda()
            future_qs = self.target_model.forward(new_states).cpu().detach().numpy()
            
            x = []
            y = []
            
            for idx, (current_state, action, reward, new_state, done) in enumerate(sample):
                # computing the bellman equation to update q-values
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
            self.running_loss += self.model.fit(torch.tensor(np.array(x), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32))
            # episode counter for updating target model
            if terminal_state:
                self.target_update_counter += 1
            # the target model update
            if self.target_update_counter > UPDATE_FREQ:
                self.target_model.load_state_dict(self.model.state_dict())
                self.target_update_counter = 0
    
    def update_replay_memory(self, transition):
        '''This just adds a game step to the replay buffer'''
        self.memory.append(transition)
    
    def build_model(self):
        model = SnakeDQN()
        return model