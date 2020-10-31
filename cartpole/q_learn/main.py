import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1')

env.reset()

LEARNING_RATE = 0.1
DISCOUNT = 0.95 # how important are future actions (weigth)
                # future reward vs current action reward.

EPISODES = 10000
SHOW_EVERY = 1000

# Due to the fact that the envirionment's observation space
# is kind of continuous we need convert it to discrete set
# of observations. 
# We have four observations and the ranges are quite (but not that much)
# diverse. I would pick 100 as a dicrete chunks number. Let's see

DISCRETE_OBSERVATION_SPACE_SIZE = [100] * len(env.observation_space.high)

print(env.observation_space.high)
print(env.observation_space.low)

# Modyfing max and min for velocities based on experiment:
env.observation_space.high[1] = 28.637816605848855 * 100
env.observation_space.low[1] = -10.053835830950502 * 100

env.observation_space.high[3] = 50.81515082112765 * 100
env.observation_space.low[3] = -0.36757979541382707 * 100

# And now let's calculate size of the chunk.
discrete_observation_space_window_size = (
    env.observation_space.high - env.observation_space.low) / DISCRETE_OBSERVATION_SPACE_SIZE

print(discrete_observation_space_window_size)

epsilon = 0.5 # parameter standing for exploration (randomness)
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2

epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# q_table = np.random.uniform(
#     low=-2, high=0, size=(DISCRETE_OBSERVATION_SPACE_SIZE + [env.action_space.n]))

ep_rewards = []

aggr_ep_rewards = {
    'ep': [],
    'avg': [],
    'min': [],
    'max': []
}


# Handling infinity problem
# Env will be run 10000 times randomly steered
# then I will take min and max of those two features.
"""
cart_velocity = []
pole_angular_velocity = []

for episode in range(EPISODES):
    done = False
    while not done:
        action = np.random.randint(0, env.action_space.n)
        new_state, reward, done, _ = env.step(action)
        print(new_state)
        cart_velocity.append(new_state[1])
        pole_angular_velocity.append(new_state[3])
        

print(f"Min velocity: {min(cart_velocity)}")
print(f"Max velocity: {max(cart_velocity)}")

print(f"Min pole ang velocity: {min(pole_angular_velocity)}")
print(f"Max pole ang velocity: {max(pole_angular_velocity)}")
env.close()
"""

# Results:
#   Min velocity: -10.053835830950502
#   Max velocity: 28.637816605848855
#   Min pole ang velocity: -0.36757979541382707
#   Max pole ang velocity: 50.81515082112765

# As a min and max, experimental values from above will
# be multiplied by 100 for algorithm
