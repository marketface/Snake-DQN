# Snake-DQN
The classic game of snake gets a Deep Q Network thrown at it.

In this project, I try my hand at implementing the Q-Learning algorithm from the field of reinforcement learning using a neural network. 

## The Environement:
The environement consists of a 50x50 array of values, each of which is coded such that it represents different components of the game. For example, 1s are the body of the snake, 2 is the apple (or whatever it is that you're trying to eat), and everything else is 0. With this, we have a fully observable representation of the game states.
## The Observations:
Observations for our agent will be 12-dimensional binary array of features for the directions of danger and food. The first 8 features indicate dangers around the snake's head. There are 8 of these because there are 8 positions *around* the snake's head. This means that the snake is very nearsighted (since the features only one directions of danger out to one unit from the head of the snake), but let's see how well the agent can do with this as a starting point. Also, the last 4 features just point to the food. If the food is at a diagonal position from the head, then 2 of these features will be active.
## The Reward Function:
This is pretty simple: +10 for each apple the snake eats or -10 if the snake dies. I tried more complicated reward functions, but this seems to do better while being simpler.
## The Model:
For an environment and observations as simple as I have outlined, I decided to use a pretty light network architecture. There are 12 input units, 2 hidden layers: the first has 256 units and the second has 128, then 4 output units for the 4 possible actions available in the action-space. The hidden layers consist of ReLu-activated units. This makes for a total of 36740 parameters. For the loss function is used the Mean-Squared-Error of observations to actions. Finally, for the optimizer, I decided to use AdamW in PyTorch since this seems to have subsumed the original Adam optimizer since the publication of the paper by Ilya Loshchilov and Frank Hutter (https://arxiv.org/abs/1711.05101) on arxiv detailing the implementation errors in the Adam optimizer.
