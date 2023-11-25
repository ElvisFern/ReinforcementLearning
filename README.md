# ReinforcementLearning

Withing this repository you will find two seperate approaches to solving the Gymnasium taxi problem. https://gymnasium.farama.org/environments/toy_text/taxi/

The first and recommended approach in order to gain an intimate understanding of Reinforcement Learning withouit needing a powerful computer id Approximate Q Learning, this method does not keep track of all the Q-values assigned at each iteration.

The second approach is good if you have some time and a powerful system to use it with, it serves as a very good entry way into deep learning if you have already understood what Reinforcement Learning is about.

## Approximate Q Learning
Below is a quick walkthrough of the code to be found for this method: 

import numpy as np: This line imports the numpy library, which is a fundamental package for scientific computing in Python. It's often abbreviated as np.

class ApproximateQLearningAgent:: This line defines a new class named ApproximateQLearningAgent. This class will represent an agent that learns how to make decisions using an approximate form of Q-learning.

The __init__ method is the constructor for the class. It initializes an instance of the ApproximateQLearningAgent class with specific parameters:

alpha: Learning rate.
gamma: Discount factor for future rewards.
num_actions: Number of possible actions.
num_states: Number of possible states.
epsilon, epsilon_decay, epsilon_min: Parameters for the epsilon-greedy policy.
self.weights = np.zeros((num_states, num_actions)): This initializes a weight matrix to zeros. Each state-action pair has a corresponding weight.

def choose_action(self, state, action_mask): This method decides which action to take given the current state and an action mask (which indicates valid actions).

Inside choose_action, it uses an epsilon-greedy policy to decide whether to explore (choose a random valid action) or exploit (choose the best-known action).

def get_q_values(self, state): This method returns the Q-values for a given state, which are simply the weights corresponding to that state.

def update(self, state, action, reward, next_state, done, action_mask): This method updates the weights based on the observed transition (state, action, reward, next_state) and whether the episode is done.

def update_epsilon(self): This method updates the epsilon value used in the epsilon-greedy policy, decaying it over time to reduce exploration.

def train_agent(episodes, alpha, gamma, env): This function trains the agent over a specified number of episodes. It takes the learning rate, discount factor, and the environment as inputs.

Inside train_agent, it iterates over each episode, resets the environment, and accumulates rewards while updating the agent's knowledge.

def test_agent(agent, env, num_episodes): This function tests the trained agent over a given number of episodes and calculates the average reward and steps per episode.


## Deep Q-Learning Using Tensorflow

class DQNAgent: - This line begins the definition of a class named DQNAgent. A class in Python is a blueprint for creating objects (a particular data structure), providing initial values for state (member variables or attributes), and implementations of behavior (member functions or methods).

def __init__(self, state_size, action_size): - This is the constructor for the DQNAgent class. It's called when an instance of the class is created. The parameters state_size and action_size are the dimensions of the state and action spaces of the environment, respectively.

3-9. The next lines initialize various attributes of the DQNAgent:

self.state_size and self.action_size store the sizes of the state and action spaces.
self.memory is an empty list used to store experiences for replay.
self.gamma is the discount factor for future rewards.
self.epsilon is the exploration rate, starting at 1 (100%).
self.epsilon_min is the minimum value that epsilon can decay to.
self.epsilon_decay is the rate at which epsilon decreases.
self.learning_rate is the learning rate for the neural network.
self.model = self._build_model() creates the neural network model by calling the _build_model method.
10-16. def _build_model(self): - This method defines the neural network architecture used for learning the Q-function:

A Sequential model is created.
Two hidden layers with 24 neurons each and ReLU activation are added.
An output layer with size self.action_size and linear activation is added.
The model is compiled with mean squared error as the loss and the Adam optimizer.
def remember(self, state, action, reward, next_state, done): - This method adds an experience tuple to the agent's memory. The tuple contains the state, action taken, reward received, the next state, and a done flag indicating if the episode has ended.
18-23. def act(self, state): - This method defines the agent's policy:

With probability self.epsilon, a random action is chosen (exploration).
Otherwise, the action with the highest predicted Q-value is chosen (exploitation).
24-41. def replay(self, batch_size): - This method performs experience replay:

If there are not enough experiences in memory, it returns immediately.
Otherwise, it samples a minibatch of experiences and uses them to update the neural network.
42-47. def one_hot_state(state): - This function is defined outside the class. It converts a state into a one-hot encoded vector.

49-57. The environment is set up using OpenAI Gym's Taxi-v3 environment. An instance of DQNAgent is created, and the number of training episodes is specified.

58-76. The main training loop:

Each episode starts by resetting the environment.
For a fixed number of steps (or until the episode ends), the agent selects actions, observes outcomes, and stores these experiences.
The agent's epsilon value is decayed over time.
If enough memories are gathered, the replay function is called to update the model.
78-91. The results of the training (rewards and losses) are plotted using Matplotlib to visualize the agent's learning progress.
