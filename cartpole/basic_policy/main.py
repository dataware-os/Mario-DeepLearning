"""

ENV docs:

    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        3       Pole Angular Velocity     -Inf                    Inf
    Actions:
        Type: Discrete(2)
        Num   Action
        0     Push cart to the left
        1     Push cart to the right
        Note: The amount the velocity that is reduced or increased is not
        fixed; it depends on the angle the pole is pointing. This is because
        the center of gravity of the pole increases the amount of energy needed
        to move the cart underneath it
    Reward:
        Reward is 1 for every step taken, including the termination step
    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]
    Episode Termination:
        Pole Angle is more than 12 degrees.
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 200.
        Solved Requirements:
        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials.
"""

import gym
import numpy as np


def basic_policy(obs):
    angle = obs[2]
    return 0 if angle < 0 else 1 # right == 1


if __name__=="__main__":
    env = gym.make('CartPole-v1')
    env.seed(55)
    obs = env.reset()

    totals = []
    for episode in range(500):
        episode_rewards = 0
        obs = env.reset()
        for step in range(200):
            action = basic_policy(obs)
            obs, reward, done, info = env.step(action)
            episode_rewards += reward
            env.render()
            if done:
                print(f"Succeeded {episode_rewards} steps")
                break
        totals.append(episode_rewards)

    print(f"mean rewards: {np.mean(totals)}")
    print(f"max no of steps: {np.max(totals)}")
