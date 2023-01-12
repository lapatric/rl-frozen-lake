# OpenAI Gym and Python for Q-learning

This repo follows the [Frozen Lake tutorial](https://www.gymlibrary.dev/environments/toy_text/frozen_lake/).

<img src="./frozenlaketrainingvis.gif" width="256">

## Environment initialisation

To initialise the Frozen Lake environment provided by OpenAI Gym we call `gym.make` as follows. These environments further provide a neat little 2D render through `env.render()`.

If `is_slippery` is set to True then the movement is partially stochastic, i.e. the agent will move in the intended direction with a probability of 1/3 and in either *perpendicular* direction with an equal probability of 1/3 in both directions.

```python
env = gym.make("FrozenLake-v1", render_mode="human", is_slippery=False)
env.reset()

env.render()
```

## Environment interaction

The following script runs a single *episode*.

```python
env.reset()
terminated = False
while(not terminated):
    # generate and apply random action
    randomAction = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(randomAction)
    print(["left", "down", "right", "up"][randomAction])
    if not terminated : time.sleep(2)

print("Episode terminated!")
```

## Agent construction

In order to learn how to get from the starting point to the destination, we need an agent we can train. It holds the knowledge of the Q-table and ultimately the learned policy. Beyond said Q-table `q_values`, the learning rate `lr`, an epsilon decay `epsilon_decay` and a discount factor `discount_factor`, we define three functions. 
- `get_action(state)`: Given a `state` we sample an action randomly (exploration) or pick the optimal action based on the current q-values (exploitation), `int(np.argmax(self.q_values[state]))`.
- `update(state, action, reward)`: This function updates the expected discounted return for taking action `action` in state `state`.
- `decay_epsilon()`: Decays the epsilon value.

The update to the *Q-table* $Q$ upon taking *action* $a$ in *state* $s$ with resulting *reward* $r$ is as follows:

$$ Q[s][a] = (1 - \lambda_{LR}) \cdot Q[s][a] +  \lambda_{LR} \cdot (r + \delta \cdot \max_{a} Q[s])$$

where $\lambda_{LR}$ is the *learning rate* and $\delta$ is the *discount factor*.

```python
class FrozenLakeAgent:
    def __init__(
        self,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95
    ):
        """
        Initialize a Reinforcement Learning agent with an empty dictionary of state-action values (q_values),
        a learning rate and an epsilon.
        Args:
        - learning_rate: Amount with which to weight newly learned reward vs old reward (1 - lr)
        - initial epsilon: The initial probability w/ with we sample random action (exploration)
        - epsilon_decay: Value by which epsilon value decays through subtraction
        - final_epsilon: Epsilon value at which decay stops
        - discount_factor: The factor by which future rewards are counted, i.e. expected return on next state (recursive)
        """
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        
        self.training_error = []
    
    def get_action(self, state: tuple[int, int, bool]) -> int:
        """
        Returns the best action with probability (1 - epsilon) -> exploitation. 
        Otherwise a random action with probability epsilon to ensure exploration.
        """
        if np.random.random() < self.epsilon:
            return env.action_space.sample()
        else:
            return int(np.argmax(self.q_values[state]))
    
    def update(
        self,
        state: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_state: tuple[int, int, bool]
    ):
        """
        Updates the Q-value of an action.
        The Q-value update is equivalent to the following weighting of old and new information by the learning rate:
        # self.q_values[state][action] = (1 - self.lr) * self.q_values[state][action] +
        #                                self.lr * (reward + self.discount_factor * future_q_value)
        The temporal difference is the difference between the old and new value over one (time) step.
        """
        future_q_value = (not terminated) * np.max(self.q_values[next_state]) 
        temporal_difference = reward + self.discount_factor * future_q_value - self.q_values[state][action]
        self.q_values[state][action] = self.q_values[state][action] + self.lr * temporal_difference
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
```

## Training

To train our agent we specify the following training attributes

- learning rate: The rate at which we scale in new information with old information.
- episodes: The number of training cycles to run.
- initial epsilon: The rate at which we explore vs exploit.
- epsilon decay: The rate by which we reduce the exploration rate.
- final epsilon: The exploration rate below which we do not go.

As can be seen in the following training script, the agent's policy (~ Q-table) is updated after each step. However, `epsilon` is only decayed after each episode (not each step).

```python
learning_rate = 0.01
n_episodes = 1000
initial_epsilon = 1.0
epsilon_decay = initial_epsilon / (n_episodes / 2)
final_epsilon = 0.1

agent = FrozenLakeAgent(
    learning_rate = learning_rate,
    initial_epsilon = initial_epsilon,
    epsilon_decay = epsilon_decay,
    final_epsilon = final_epsilon,
)

env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)
for episode in tqdm(range(n_episodes)):
    state, info = env.reset()
    done = False
    
    #play one episode
    while not done:
        action = agent.get_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)

        # update the agent
        agent.update(state, action, reward, terminated, next_state)

        # update done status and state
        done = terminated or truncated
        state = next_state

    # once a game is finished we decay epsilon -> converge towards exploitation
    agent.decay_epsilon()
```
The [gif](#openai-gym-and-python-for-q-learning) at the very top of this README is a visualisation of the agent learning to find the gift (target) over 500 episodes. 
