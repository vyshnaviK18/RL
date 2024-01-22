# RL


1
import numpy as np

grid_size = (3, 4)
cheese_positions = [(0, 1), (1, 3), (2, 0)]

grid_rewards = np.zeros(grid_size)

for pos in cheese_positions:
    grid_rewards[pos] = 1

value_function = np.copy(grid_rewards)

num_iterations = 100
discount_factor = 0.9

for _ in range(num_iterations):
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            if (i, j) not in cheese_positions:
                value_function[i, j] = grid_rewards[i, j] + discount_factor * np.max([
                    value_function[max(0, i-1), j],
                    value_function[min(grid_size[0]-1, i+1), j],
                    value_function[i, max(0, j-1)],
                    value_function[i, min(grid_size[1]-1, j+1)]
                ])

print("Optimal value function:")
print(value_function)

2

import numpy as np

grid_size = (5, 5)
fire_positions = [(1, 4)]
max_reward_position = (2, 4)
initial_position = (1, -1)

grid_rewards = np.zeros(grid_size)
grid_rewards[fire_positions] = -1
grid_rewards[max_reward_position] = 1

value_function = np.copy(grid_rewards)

num_iterations = 100
discount_factor = 0.9

for _ in range(num_iterations):
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            if (i, j) != max_reward_position and (i, j) not in fire_positions:
                value_function[i, j] = grid_rewards[i, j] + discount_factor * np.max([
                    value_function[max(0, i-1), j],
                    value_function[min(grid_size[0]-1, i+1), j],
                    value_function[i, max(0, j-1)],
                    value_function[i, min(grid_size[1]-1, j+1)]
                ])

print("Optimal value function:")
print(value_function)

3

import numpy as np

grid_size = (6, 4)
initial_position = (0, 0)

grid_rewards = np.random.rand(*grid_size)  # Random rewards for each cell
value_function = np.zeros(grid_size)

num_iterations, discount_factor, epsilon = 100, 0.9, 0.1

for _ in range(num_iterations):
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            if (i, j) != initial_position:
                explore_exploit = np.random.choice([0, 1], p=[epsilon, 1-epsilon])

                if explore_exploit:  # Exploit
                    value_function[i, j] = np.max([
                        value_function[max(0, i-1), j],
                        value_function[min(grid_size[0]-1, i+1), j],
                        value_function[i, max(0, j-1)],
                        value_function[i, min(grid_size[1]-1, j+1)]
                    ])
                else:  # Explore
                    value_function[i, j] = grid_rewards[i, j] + discount_factor * np.max([
                        value_function[max(0, i-1), j],
                        value_function[min(grid_size[0]-1, i+1), j],
                        value_function[i, max(0, j-1)],
                        value_function[i, min(grid_size[1]-1, j+1)]
                    ])

print("Optimal value function:")
print(value_function)



4



import numpy as np

num_states, num_actions = 5, 3
Q_values = np.zeros((num_states, num_actions))

def update_q_values(episode, gamma=0.9):
    for i in range(len(episode)):
        state, action, reward = episode[i]
        future_rewards = sum([gamma**j * episode[i+j][2] for j in range(len(episode)-i-1)])
        Q_values[state, action] += future_rewards

for _ in range(10):
    episode = [(0, np.random.choice(num_actions), np.random.uniform(-1, 1)) for _ in range(10)]
    update_q_values(episode)

    total_reward = sum([step[2] for step in episode])
    print(f"Total Reward: {total_reward}")

print("\nLearned Q-values:")
print(Q_values)







1





import torch
import torch.nn as nn
import torch.optim as optim
import gym

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(nn.Linear(input_size, 128), nn.ReLU(), nn.Linear(128, output_size), nn.Softmax(dim=-1))

    def forward(self, x):
        return self.fc(x)

env, agent, num_episodes = gym.make('CartPole-v1'), PolicyNetwork(4, 2), 1000

for episode in range(num_episodes):
    state, total_reward = env.reset(), 0

    while True:
        action = torch.distributions.Categorical(agent(torch.tensor(state, dtype=torch.float32))).sample().item()
        next_state, reward, done, _ = env.step(action)

        loss = -torch.log(agent(torch.tensor(state, dtype=torch.float32))[action]) * reward
        agent.fc.zero_grad(), loss.backward(), optim.Adam(agent.parameters(), lr=0.01).step()

        total_reward, state = total_reward + reward, next_state

        if done: break

    if episode % 10 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}")

env.close()

