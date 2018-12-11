import os
import sys
from argparse import ArgumentParser
from itertools import zip_longest
import tempfile
from datetime import datetime
import shutil
import logging

import gym
import billiard_ai
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
from tensorboardX import SummaryWriter
import yaml

import utils

# Verbosity
logging.basicConfig(level=logging.INFO)

# Environment constants
NUM_INPUT_CHANNELS = 3
NUM_ACTION_DIMS = 2
HEIGHT, WIDTH = 900, 600
MAX_SCORE = 22/4

# TODO: Abortion criterion
# TODO: Multiple actions?
# TODO: First state assumption
# TODO: Sensible parameters
# TODO: Animation
# TODO: Limit velocity


def build_convolutional_network(in_channels, out_channels,
                                kernel_sizes, strides,
                                in_height, in_width):
    cnn = []
    h, w = in_height, in_width
    for in_channel, out_channel, kernel_size, stride in zip(in_channels, out_channels, kernel_sizes, strides):
        # Spatial dimensions
        h = utils.compute_output_shape(h_in=h,
                                       kernel_size=kernel_size[0],
                                       stride=stride[0])
        w = utils.compute_output_shape(h_in=w,
                                       kernel_size=kernel_size[1],
                                       stride=stride[1])

        # Layer
        cnn.append(nn.Conv2d(in_channels=in_channel,
                             out_channels=out_channel,
                             kernel_size=kernel_size,
                             stride=stride))

    # Spatial dimensions of last channels
    last_h, last_w = h, w

    return cnn, last_h, last_w


def build_dense_network(num_units):
    fc = []
    for in_features, out_features in zip(num_units[:-1], num_units[1:]):
        fc.append(nn.Linear(in_features=in_features,
                            out_features=out_features))
    return fc


class ActorNetwork(nn.Module):

    def __init__(self, config):
        super().__init__()

        # Copy
        self.cnn_activations = list(map(utils.resolve_activations, config["cnn_activations"]))
        self.cnn_kernel_sizes = config["cnn_kernel_sizes"]
        self.cnn_strides = config["cnn_strides"]
        self.cnn_in_channels = [NUM_INPUT_CHANNELS] + config["cnn_num_channels"]
        self.cnn_out_channels = config["cnn_num_channels"] + [NUM_ACTION_DIMS]

        # Convolutional layers
        self.cnn, w, h = build_convolutional_network(in_channels=self.cnn_in_channels,
                                                     out_channels=self.cnn_out_channels,
                                                     kernel_sizes=self.cnn_kernel_sizes, strides=self.cnn_strides,
                                                     in_height=HEIGHT, in_width=WIDTH)

        # Dense layers
        in_features = self.cnn_out_channels[-1] * h * w
        self.fc_num_layers = config["fc_num_layers"] + 1
        self.fc_activations = list(map(utils.resolve_activations, config["fc_activations"]))
        self.fc_num_units = [in_features] + config["fc_num_units"] + [3]
        self.fc = build_dense_network(num_units=self.fc_num_units)

        # Make PyTorch aware of sub-networks
        self.cnn = nn.ModuleList(self.cnn)
        self.fc = nn.ModuleList(self.fc)

    def forward(self, x):

        # Pass through convolutional layers
        for cnn_layer, activation in zip(self.cnn, self.cnn_activations):
            x = activation()(cnn_layer(x))

        # Flatten last two dimensions
        x = x.view(x.shape[0], -1)  # [B, C * H * W]

        # Pass through fully-connected layers
        for fc_layer, activation in zip_longest(self.fc, self.fc_activations):
            x = fc_layer(x)
            if activation:
                x = activation()(x)

        # Apply suitable non-linearities
        angle_uv = torch.nn.Softsign()(x[:, :2])  # in (-1, +1)
        velocity = torch.nn.Softplus()(x[:, 2])  # in (0, +inf)

        # # Derive angle
        angle = torch.atan2(angle_uv[:, 0], angle_uv[:, 1])

        return torch.cat((angle.unsqueeze(dim=1), velocity.unsqueeze(dim=1)), dim=1)


class CriticNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Copy
        self.state_cnn_activations = list(map(utils.resolve_activations, config["state_cnn_activations"]))
        self.state_cnn_kernel_sizes = config["state_cnn_kernel_sizes"]
        self.state_cnn_strides = config["state_cnn_strides"]
        self.state_cnn_in_channels = [NUM_INPUT_CHANNELS] + config["state_cnn_num_channels"][:-1]
        self.state_cnn_out_channels = config["state_cnn_num_channels"]

        """
        STATE
        """

        # Convolutional network
        self.state_cnn, w, h = build_convolutional_network(in_channels=self.state_cnn_in_channels,
                                                           out_channels=self.state_cnn_out_channels,
                                                           kernel_sizes=self.state_cnn_kernel_sizes,
                                                           strides=self.state_cnn_strides,
                                                           in_height=HEIGHT, in_width=WIDTH)

        """
        STATE & ACTION
        """

        # Dense layer
        in_features = self.state_cnn_out_channels[-1] * h * w + NUM_ACTION_DIMS
        self.fc_num_layers = config["fc_num_layers"] + 1
        self.fc_activations = list(map(utils.resolve_activations, config["fc_activations"]))
        self.fc_num_units = [in_features] + config["fc_num_units"] + [1]
        self.fc = build_dense_network(num_units=self.fc_num_units)

        # Make PyTorch aware of sub-networks
        self.state_cnn = nn.ModuleList(self.state_cnn)
        self.fc = nn.ModuleList(self.fc)

    def forward(self, state, action):

        """ STATE """

        # Pass through convolutional layers
        x = state
        for cnn_layer, activation in zip(self.state_cnn, self.state_cnn_activations):
            x = activation()(cnn_layer(x))

        # Flatten last two dimensions
        x = x.view(x.shape[0], -1)  # [B, C * H * W]

        """ STATE & ACTION """

        # Combine with action
        x = torch.cat((x, action), dim=1)

        # Pass through fully-connected layers
        for fc_layer, activation in zip_longest(self.fc, self.fc_activations):
            x = fc_layer(x)
            if activation:
                x = activation()(x)

        return x


def test_agent(env, actor, device):

    state, reward, done, _ = env.reset()

    while not done:
        env.render()
        action = actor(torch.tensor(state, device=device).unsqueeze(dim=0)).squeeze(dim=0)
        state, reward, done, _ = env.step(action.cpu().detach().numpy())


def main():
    # Instantiate parser
    parser = ArgumentParser()

    parser.add_argument("--network_config",
                        help="Definition of the network",
                        required=True)

    parser.add_argument("--test",
                        help="Whether the test mode should be activated",
                        action="store_true",
                        default=False)

    # Parse
    args = parser.parse_args()

    # Input assertions
    assert os.path.exists(args.network_config), "Network configuration does not exist!"

    # Read config
    with open(args.network_config, "r") as f:
        try:
            config = yaml.load(f)
        except yaml.YAMLError as e:
            print(e)
            sys.exit(1)

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Build Gym environment
    env = gym.make('billiard-v0')

    # Build networks
    actor = ActorNetwork(config["actor"]).to(device)
    critic = CriticNetwork(config["critic"]).to(device)

    if args.test:
        # Restore actor's weights
        actor.load_state_dict(torch.load(os.path.join(os.path.dirname(args.network_config), "actor.pt")))
        test_agent(env, actor, device)
        return

    # Temporary files
    tmp_dir = os.path.join(tempfile.gettempdir(), "PyTorch_{date:%Y-%m-%d_%H_%M_%S}".format(date=datetime.now()))
    os.makedirs(tmp_dir)
    shutil.copyfile(src=args.network_config, dst=os.path.join(tmp_dir, os.path.basename(args.network_config)))
    logging.info("Temporary directory: {}".format(tmp_dir))

    # Build target networks
    actor_target = ActorNetwork(config["actor"]).to(device)
    critic_target = CriticNetwork(config["critic"]).to(device)

    # Enforce same weights
    actor_target.load_state_dict(actor.state_dict())
    critic_target.load_state_dict(critic.state_dict())

    # Replay buffer
    replay_buffer = utils.ReplayBuffer(buffer_size=config["replay_buffer_size"])

    # Optimizers
    actor_optimizer = optim.RMSprop(actor.parameters(), lr=config["learning_rate"])
    critic_optimizer = optim.RMSprop(critic.parameters(), lr=config["learning_rate"])

    # Noise levels
    noise_loc = torch.tensor([0., 0.], dtype=torch.float32, device=device)
    noise_std = torch.tensor([config["noise_std_angle"], config["noise_std_velocity"]],
                             dtype=torch.float32, device=device)
    noise = distributions.Normal(loc=noise_loc, scale=noise_std)

    # TensorBoard
    summary_writer = SummaryWriter(os.path.join(tmp_dir))

    # Training loop
    state, reward, done, _ = env.reset()

    i = 0
    global_step = 0
    while global_step < 10:
        # Pass through actor
        action = actor(torch.tensor(state, device=device).unsqueeze(dim=0))

        # Add noise
        action = action + noise.sample()

        # Renormalize (values could fall out of range)
        action = action.squeeze()
        action[0] = torch.atan2(action[0].sin(), action[0].cos())
        action[1] = nn.Softplus()(action[1])

        # Perform action
        action = action.cpu().detach().numpy()
        new_state, reward, done, _ = env.step(action=action)

        # Add to replay buffer
        replay_buffer.add_transition((state, action, reward, new_state, done))

        # Increment global counter
        i += 1

        # Reset environment if necessary
        if done:
            state, reward, done, _ = env.reset()
        else:
            state = new_state

        if i == config["update_networks_every_n_steps"]:
            for j in range(config["update_networks_for_n_steps"]):

                # Obtain batch
                state_batch, action_batch, reward_batch, new_state_batch, done_batch = replay_buffer.get_batch(
                    batch_size=config["batch_size"])

                # Convert to tensors
                state_batch = torch.tensor(state_batch, device=device)
                action_batch = torch.tensor(action_batch, device=device)
                reward_batch = torch.tensor(reward_batch, device=device)
                new_state_batch = torch.tensor(new_state_batch, device=device)
                done_batch = torch.tensor(done_batch, device=device)

                # Targets
                q_target = critic_target(new_state_batch, actor_target(new_state_batch)).squeeze(dim=1)
                rhs = reward_batch + config["gamma"] * (1 - done_batch) * q_target

                # Critic gradients
                critic_optimizer.zero_grad()
                q = critic(state_batch, action_batch).squeeze(dim=1)
                critic_loss = nn.MSELoss()(q, rhs)
                critic_loss.backward()

                # Actor gradients
                actor_optimizer.zero_grad()
                actor_performance = -torch.mean(critic(state_batch, actor(state_batch)))
                actor_performance.backward()

                # Perform steps
                critic_optimizer.step()     # descent
                actor_optimizer.step()      # ascent

                # TensorBoard
                summary_writer.add_scalar(tag="critic",
                                          scalar_value=critic_loss.cpu().detach().numpy(),
                                          global_step=global_step)
                summary_writer.add_scalar(tag="actor",
                                          scalar_value=actor_performance.cpu().detach().numpy(),
                                          global_step=global_step)

                # Increment step counter
                global_step += 1

            # Copy updated weights into target networks
            actor_target.load_state_dict(actor.state_dict())
            critic_target.load_state_dict(critic.state_dict())

            # Reset global counter
            i = 0

    # Save weights
    torch.save(actor.state_dict(), os.path.join(tmp_dir, "actor.pt"))
    torch.save(critic.state_dict(), os.path.join(tmp_dir, "critic.pt"))


if __name__ == "__main__":
    main()
