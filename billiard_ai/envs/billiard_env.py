import gym
from gym import spaces
import numpy as np
import billiard


# Game constants
SCREEN_WIDTH, SCREEN_HEIGHT = 600, 900
POCKET_WIDTH, BALL_RADIUS = 40, 10
NUM_BALLS = 22
FPS = 60
V_MAX = 500


class BilliardEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):

        # Instantiate game
        self.game = billiard.BilliardGame(
            screen_width=SCREEN_WIDTH, screen_height=SCREEN_HEIGHT,
            pocket_width=POCKET_WIDTH, ball_radius=BALL_RADIUS,
            num_balls=NUM_BALLS, fps=FPS
        )

        # Gym-specific
        self.action_space = spaces.Tuple((
            spaces.Box(low=0, high=360, shape=(1,), dtype=np.float32),
            spaces.Box(low=0, high=V_MAX, shape=(1,), dtype=np.float32))
        )
        self.observation_space = spaces.Box(low=0, high=1., shape=[SCREEN_HEIGHT, SCREEN_WIDTH], dtype=np.float32)
        self.is_initialized = True

    def step(self, action):
        assert self.is_initialized, "Env. initialization required!"
        assert not self.game.done, "Reset required"
        _step = self.game.step(phi_deg=action["phi_deg"],
                               velocity_magnitude=action["velocity_magnitude"])
        return _step.new_observation, _step.reward, _step.done, {}

    def reset(self):
        assert self.is_initialized, "Env. initialization required!"
        _step = self.game.reset()
        return _step.new_observation, _step.reward, _step.done, {}

    def render(self, mode='human'):
        pass

