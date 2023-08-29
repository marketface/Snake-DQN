# Snake-DQN

For this project, I'll be using PyTorch as the neural network library of choice.

### Environment

The environment consists of a 50x50 array of values, each of which is coded such that it represents different components of the game. For example, 1s are the body of the snake, 2 is the apple (or whatever it is the snake eats), and everything else is 0. With this, I can have a full representation of the game states in a single, concise array.

### Observations

Observations for our agent will consist of a 12-dimensional binary array of features for the directions of danger and food. The first 8 features indicate dangers around the snake's head. The last 4 features just point to the food. If the food is at a diagonal position from the head, then 2 of these features will be active.

### Reward Function:

 + 1 * (length of the snake) - 3 where I am subtracting 3 for the starting length of the snake or -1 if the snake dies.

### Model

For an environment and observations as simple as I have outlined, I decided to use a pretty light network architecture. There are 12 input units and 2 hidden layers: the first has 256 units and the second has 128, then 4 output units for the 4 possible actions available in the action-space. Leaky ReLu was used as the activation function across all but output layers. This makes for a total of 36740 parameters. 

Model loss was calculated using mean-squared error optimized using AdamW.

## Training

For training, I used an experience replay buffer to sample random state-action pairs to use for training in a supervised setting. This a standard approach to utilizing neural networks in reinforcement learning applications since it decorrelates the state-action pairs from one another, rendering them (closer) to i.i.d.

To train the model, I attempted to follow the paper, *Deep Reinforcement Learning with Double Q-learning*, by van Hasselt et al (https://arxiv.org/pdf/1509.06461.pdf). This entails training two networks at once (another reason for the light network architecture): a target network and a training network. The training network is used to generate current guesses for the q-values: it is updated at every timestep. The target network update frequency is a tunable hyperparameter that I messed around with. I settled on updating the target network every 7500 timesteps. Doing this stabilizes the predictions of future q-values when used to optimize the training network.

I trained the network for a total of 10000 episodes overnight (~ 9 hours of training), below are the results.

## Results

Here is the agent, after 9200 games:


https://github.com/steez-ml/Snake-DQN/assets/39159387/40b0affc-9f41-4f41-9b95-b9eae7f8fafc

