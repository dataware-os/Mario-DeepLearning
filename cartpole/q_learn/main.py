# %%
import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1')

env.reset()

DEBUG = False

LEARNING_RATE = 0.1
DISCOUNT = 0.99 # how important are future actions (weigth)
                # future reward vs current action reward.

EPISODES = 100000
SHOW_EVERY = 10000

# Due to the fact that the envirionment's observation space
# is kind of continuous we need convert it to discrete set
# of observations.
# We have four observations and the ranges are quite (but not that much)
# diverse. I would pick CHUNKS as a dicrete chunks number. Let's see

CHUNKS = 100

DISCRETE_OBSERVATION_SPACE_SIZE = [CHUNKS] * len(env.observation_space.high)

# %%
print(env.observation_space.high)
print(env.observation_space.low)

# Modyfing max and min for velocities based on experiment
# (avg taken from observation):
env.observation_space.high[1] = 10.0
env.observation_space.low[1] = -10.0

env.observation_space.high[3] = 10.0
env.observation_space.low[3] = -10.0

# And now let's calculate size of the chunk.
discrete_observation_space_window_size = (
    env.observation_space.high - env.observation_space.low) / DISCRETE_OBSERVATION_SPACE_SIZE

# %%
print(discrete_observation_space_window_size)

epsilon = 0.5 # parameter standing for exploration (randomness)
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2

epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

q_table = np.random.uniform(
    low=0, high=1, size=(DISCRETE_OBSERVATION_SPACE_SIZE + [env.action_space.n]))

ep_rewards = []

aggr_ep_rewards = {
    'ep': [],
    'avg': [],
    'min': [],
    'max': []
}

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_observation_space_window_size
    return tuple(discrete_state.astype(np.int))

print("Started learning")
for episode in range(EPISODES):
    print_debug(f"Episode: {episode}")

    done = False
    episode_reward = 0

    discrete_state = get_discrete_state(env.reset())
    if episode % SHOW_EVERY == 0:
        print_debug(f"Rendering episode: {episode}")
        render = True
    else:
        render = False

    while not done:

        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
            print_debug(f"Chosen action: {action}")
        else:
            print_debug(f"Chosen random action (exploring): {action}")
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, done, _ = env.step(action)

        print_debug(f"new_state, reward, done: {new_state}, {reward}, {done}")

        episode_reward += reward
        new_discrete_state = get_discrete_state(new_state)

        print_debug(f"New discrete state: {new_discrete_state}")

        if render:
            env.render()

        if any(x >= CHUNKS for x in new_discrete_state):
            done = True
            continue

        max_future_q = np.max(q_table[new_discrete_state])
        current_q = q_table[discrete_state + (action, )]
        new_q = (1-LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        pos_in_arr = discrete_state + (action,)

        print_debug(f"Current q: {current_q}")
        print_debug(f"New q: {new_q}")
        print_debug(f"position in arr: {discrete_state + (action,)}")

        q_table[pos_in_arr] = new_q # we update q table after taking an action...

        if episode_reward >= 200:
            print_debug(f'Already success: {episode}')

        elif episode_reward >= 400:
            done = True

        discrete_state = new_discrete_state

    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

    ep_rewards.append(episode_reward)

    if not episode % SHOW_EVERY:
        average_reward = sum(ep_rewards[-SHOW_EVERY:]) / len(ep_rewards[-SHOW_EVERY:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))

        print(f'Episode: {episode}, avg: {average_reward}, min: {min(ep_rewards[-SHOW_EVERY:])}, max: {max(ep_rewards[-SHOW_EVERY:])}')

env.close()

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="avg")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max")

plt.legend(loc=4)
plt.show()


def print_debug(message):
    if DEBUG:
        print(message)


"""
Code converged after ~60 000 episodes (avg reached 500)

Started learning
0
Episode: 0, avg: 16.0, min: 16.0, max: 16.0
10000
Episode: 10000, avg: 46.7428, min: 8.0, max: 246.0
20000
Episode: 20000, avg: 78.9743, min: 8.0, max: 432.0
30000
Episode: 30000, avg: 123.4969, min: 9.0, max: 500.0
40000
Episode: 40000, avg: 241.333, min: 10.0, max: 500.0
50000
Episode: 50000, avg: 445.8528, min: 16.0, max: 500.0
60000
Episode: 60000, avg: 498.1377, min: 112.0, max: 500.0
70000
Episode: 70000, avg: 500.0, min: 500.0, max: 500.0
80000
Episode: 80000, avg: 499.9606, min: 106.0, max: 500.0
90000
Episode: 90000, avg: 500.0, min: 500.0, max: 500.0

"""
