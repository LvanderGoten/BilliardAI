from gym.envs.registration import register

register(
    id='billiard-v0',
    entry_point='billiard_ai.envs:BilliardEnv',
    nondeterministic=True
)
