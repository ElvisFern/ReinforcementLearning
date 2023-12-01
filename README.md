

## Approximate Q Learning
----------------------------
### Learning Approach:

*	Linear function approximation with weights for estimating Q-values.
*	Key parameters: learning rate (alpha), discount rate (gamma), exploration rate (epsilon), and weight matrix for Q-values
*	Methods for action selection (choose_action), updating Q-values (update), and updating exploration rate (update_epsilon).

### Evaluation Metrics:

* Training Phase: Rewards are accumulated per episode during training. The focus is on how the agent's policy improves over time.
* Testing Phase: The agent is tested over a set number of episodes, calculating the average reward and average steps per episode which lets us know how robust the policy is learned

### Visualization
*	A single plot is generated to visualize the average rewards per episode during the training phase. This plot helps in understanding the learning progression and effectiveness of the agent's policy over time.



The __init__ method is the constructor for the class. It initializes an instance of the ApproximateQLearningAgent class with specific parameters:

alpha: Learning rate.
gamma: Discount factor for future rewards.
num_actions: Number of possible actions.
num_states: Number of possible states.
epsilon, epsilon_decay, epsilon_min: Parameters for the epsilon-greedy policy.
self.weights = np.zeros((num_states, num_actions)): This initializes a weight matrix to zeros. Each state-action pair has a corresponding weight.

![image](https://github.com/ElvisFern/ReinforcementLearning/assets/78712154/5bca9c7f-e6bb-4615-951f-9375a0b41745)


def choose_action(self, state, action_mask): This method decides which action to take given the current state and an action mask (which indicates valid actions).

Inside choose_action, it uses an epsilon-greedy policy to decide whether to explore (choose a random valid action) or exploit (choose the best-known action).
![image](https://github.com/ElvisFern/ReinforcementLearning/assets/78712154/69bf940c-b51c-42db-8a40-0527f049d89f)


def get_q_values(self, state): This method returns the Q-values for a given state, which are simply the weights corresponding to that state.
![image](https://github.com/ElvisFern/ReinforcementLearning/assets/78712154/a116fe0c-60e2-4a76-bb39-e18570101911)


def update(self, state, action, reward, next_state, done, action_mask): This method updates the weights based on the observed transition (state, action, reward, next_state) and whether the episode is done.
![image](https://github.com/ElvisFern/ReinforcementLearning/assets/78712154/58d0a7bc-a886-4a8c-ab4d-61bbd3ddda6c)


def update_epsilon(self): This method updates the epsilon value used in the epsilon-greedy policy, decaying it over time to reduce exploration.
![image](https://github.com/ElvisFern/ReinforcementLearning/assets/78712154/5c17c1f0-bb7f-4f6b-9b1c-23a732ebb27d)


def train_agent(episodes, alpha, gamma, env): This function trains the agent over a specified number of episodes. It takes the learning rate, discount factor, and the environment as inputs.

Inside train_agent, it iterates over each episode, resets the environment, and accumulates rewards while updating the agent's knowledge.

![image](https://github.com/ElvisFern/ReinforcementLearning/assets/78712154/a83efbf3-b143-4ec4-925a-7dbedc09f4f2)


def test_agent(agent, env, num_episodes): This function tests the trained agent over a given number of episodes and calculates the average reward and steps per episode.
![image](https://github.com/ElvisFern/ReinforcementLearning/assets/78712154/bad4cd3f-31de-43f5-97e9-45a044f9bf5f)


## Deep Q-Learning Using Tensorflow
------------------------------------

### Learning Approach:
* A neural network model is built for function approximation. The network consists of two hidden layers with 24 neurons each, using ReLU activation. The output layer has a size equal to the number of actions and uses linear activation.
* Key parameters: discount rate (gamma), exploration rate (epsilon), learning rate, and a memory buffer for experience replay.
* Methods for choosing actions (act), storing experiences (remember), and learning from experiences (replay).

### Evaluation Metrics:
* The performance of the agent is evaluated based on the rewards per episode and the loss during training.
* The agent is trained over multiple episodes, with rewards accumulated per episode. Additionally, the average loss during replay (experience replay) is calculated.
* There is also a test phase where the trained agent is evaluated over a number of episodes to compute the average test reward in order to test the robustness of the learned policy.

### Visualization :
* Training Rewards per Episode: This plot visualizes the total reward accumulated in each training episode, providing insight into how the agent's performance improves over time.
* Training Loss per Replay: This plot shows the loss incurred during each replay (experience replay), indicating how well the agent is learning from past experiences.



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

![image](https://github.com/ElvisFern/ReinforcementLearning/assets/78712154/5f62187c-314d-4ea7-a6f4-09c7562aa5c1)

def _build_model(self): - This method defines the neural network architecture used for learning the Q-function:

A Sequential model is created.
Two hidden layers with 24 neurons each and ReLU activation are added.
An output layer with size self.action_size and linear activation is added.
The model is compiled with mean squared error as the loss and the Adam optimizer.
![image](https://github.com/ElvisFern/ReinforcementLearning/assets/78712154/7c4cb66c-ff65-40db-875d-57505d465c1c)


def remember(self, state, action, reward, next_state, done): - This method adds an experience tuple to the agent's memory. The tuple contains the state, action taken, reward received, the next state, and a done flag indicating if the episode has ended.
![image](https://github.com/ElvisFern/ReinforcementLearning/assets/78712154/86dcc2a0-913f-43dc-b327-a149c28fe551)


def act(self, state): - This method defines the agent's policy:

With probability self.epsilon, a random action is chosen (exploration).
Otherwise, the action with the highest predicted Q-value is chosen (exploitation).
![image](https://github.com/ElvisFern/ReinforcementLearning/assets/78712154/d2b18f3d-8441-48e5-903e-defe320deaba)

def replay(self, batch_size): - This method performs experience replay:

If there are not enough experiences in memory, it returns immediately.
Otherwise, it samples a minibatch of experiences and uses them to update the neural network.
![image](https://github.com/ElvisFern/ReinforcementLearning/assets/78712154/0c8f6c2f-7b6a-4119-8d2a-04f8f5bec917)



 def one_hot_state(state): - This function is defined outside the class. It converts a state into a one-hot encoded vector.

 ![image](https://github.com/ElvisFern/ReinforcementLearning/assets/78712154/076b48d5-1edf-4961-bac7-9c244be71ae6)





![image](https://github.com/ElvisFern/ReinforcementLearning/assets/78712154/d1f24160-728d-481f-8fbb-392d64f66b30)

The results of the training (rewards and losses) are plotted using Matplotlib to visualize the agent's learning progress.
![image](https://github.com/ElvisFern/ReinforcementLearning/assets/78712154/729f39bf-1558-4585-b4b4-e0a1003d4ea0)
