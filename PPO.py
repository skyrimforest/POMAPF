'''
@Project ：JiangPro 
@File    ：PPO.py
@IDE     ：PyCharm 
@Author  ：Skyrim
@Date    ：2025/6/17 9:33 
'''

import torch
import torch.nn as nn
from pogema import GridConfig, pogema_v0
import numpy as np


class ActorCritic(nn.Module):
    def __init__(self, input_shape=(3, 7, 7), n_actions=5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        conv_out_size = 32 * 7 * 7
        self.actor = nn.Sequential(
            nn.Linear(conv_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
        self.critic = nn.Sequential(
            nn.Linear(conv_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.actor(x), self.critic(x)

    def act(self, x):
        logits, _ = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()


class PPOAgent:
    def __init__(self, device='cpu'):
        self.model = ActorCritic().to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        self.trajectory = []

    def act(self, obs_tensor, valid_actions):
        # 将 3x7x7 的 obs_tensor 转为 tensor 并送入网络
        obs = torch.tensor(obs_tensor, dtype=torch.float32).unsqueeze(0).to(self.device)
        action, log_prob, entropy = self.model.act(obs)
        # 简单策略：如果 action 无效，就随机选一个 valid 的
        if action not in valid_actions:
            action = np.random.choice(valid_actions)
        self.trajectory.append((obs, action, log_prob))
        return action

    def store(self, reward, done):
        self.trajectory[-1] += (reward, done)

    def train(self):
        # 等收集一段 trajectory 后批量训练
        # 伪代码结构，具体实现需 GAE/advantage 等
        obs, actions, log_probs, rewards = [], [], [], []
        for o, a, lp, r, d in self.trajectory:
            obs.append(o)
            actions.append(torch.tensor([a], dtype=torch.int64))
            log_probs.append(lp)
            rewards.append(torch.tensor([r], dtype=torch.float32))
        # 示例训练代码（省略 advantage）
        obs = torch.cat(obs)
        actions = torch.cat(actions)
        old_log_probs = torch.stack(log_probs)
        returns = torch.cat(rewards)

        logits, values = self.model(obs)
        new_probs = torch.softmax(logits, dim=1)
        dist = torch.distributions.Categorical(new_probs)
        new_log_probs = dist.log_prob(actions)

        ratio = (new_log_probs - old_log_probs).exp()
        surr1 = ratio * returns
        surr2 = torch.clamp(ratio, 0.8, 1.2) * returns
        loss = -torch.min(surr1, surr2).mean() + (returns - values.squeeze()).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.trajectory.clear()

# 初始化环境
seed = 43
env = pogema_v0(GridConfig(integration="PettingZoo", num_agents=4, obs_radius=3, seed=seed))
observations = env.reset(seed=seed)
env.render()

agent_map = {agent: PPOAgent() for agent in env.agents}

while not all(done.values()):
    actions = {}
    for agent in env.agents:
        if done[agent]: continue
        obs_tensor = observations[agent]
        actions[agent] = agent_map[agent].act(obs_tensor, valid_actions)

    observations, rewards, terminations, truncations, infos = env.step(actions)

    for agent in env.agents:
        agent_map[agent].store(rewards[agent], terminations[agent] or truncations[agent])

    # 每 N 步或 N episode 后调用
    for agent in env.agents:
        agent_map[agent].train()
