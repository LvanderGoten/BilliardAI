import gym
import billiard_ai
from skimage import io
import matplotlib.pyplot as plt
env = gym.make('billiard-v0')
df = env.reset()
res = env.step({"phi_deg": 45, "velocity_magnitude": 300})
io.imshow(res[0])
plt.show()