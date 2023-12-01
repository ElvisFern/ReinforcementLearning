# The Problem Space

Taxi-v3 from the gym series innitially started with openAI currently being continued by [Gymnasium](https://gymnasium.farama.org/environments/toy_text/taxi/)

### State Space :
Discrete 500 

### Action Space:
Discrete 6

For more details navigate to the [Taxi](https://gymnasium.farama.org/environments/toy_text/taxi/) problem page by clicking on the hyperlinks.
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





