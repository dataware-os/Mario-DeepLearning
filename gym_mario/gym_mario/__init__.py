from gym.envs.registration import register
 
register(id='Mario-v0', 
    entry_point='gym_mario.envs:MarioEnv', 
)