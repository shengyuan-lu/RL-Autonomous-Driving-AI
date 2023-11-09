import gym
import carla
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(self._get_conv_output(input_shape), 512)
        self.actor = nn.Linear(512, num_actions)
        self.critic = nn.Linear(512, 1)

    def _get_conv_output(self, shape):
        o = self.conv1(torch.zeros(1, *shape))
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))

    def forward(self, x):
        x = x.float() / 255
        x = x.permute(0, 3, 1, 2)
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.relu(self.fc1(x.view(x.size(0), -1)))
        return self.actor(x), self.critic(x)

class PPO():
    def __init__(self, input_shape, num_actions, lr=0.0003, gamma=0.99, clip_ratio=0.2, ppo_epochs=10, batch_size=64):
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.lr = lr
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size

        self.actor_critic = ActorCritic(input_shape, num_actions)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)

    def act(self, state):
        state = torch.from_numpy(state).unsqueeze(0)
        with torch.no_grad():
            dist, _ = self.actor_critic(state)
            dist = Categorical(dist)
            action = dist.sample()
        return action.item()

    def evaluate(self, state, action):
        dist, value = self.actor_critic(state)
        dist = Categorical(dist)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy().mean()
        return log_prob, value, entropy

    def update(self, states, actions, log_probs, returns, advantages):
        for _ in range(self.ppo_epochs):
            for state, action, old_log_prob, return_, advantage in self._get_batches(states, actions, log_probs, returns, advantages):
                log_prob, value, entropy = self.evaluate(state, action)
                ratio = (log_prob - old_log_prob).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantage
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = (return_ - value).pow(2).mean()
                entropy_loss = -entropy.mean()
                loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def _get_batches(self, states, actions, log_probs, returns, advantages):
        batch_size = len(states) // self.batch_size
        for i in range(batch_size):
            idx = slice(i * self.batch_size, (i + 1) * self.batch_size)
            yield states[idx], actions[idx], log_probs[idx], returns[idx], advantages[idx]

    def save(self, path):
        torch.save(self.actor_critic.state_dict(), path)

    def load(self, path):
        self.actor_critic.load_state_dict(torch.load(path))
        self.actor_critic.eval()