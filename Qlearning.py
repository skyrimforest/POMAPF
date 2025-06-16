'''
@Project ：JiangPro
@File    ：HybridAgent.py
@IDE     ：PyCharm
@Author  ：Skyrim
@Date    ：2025/5/9
'''

import numpy as np
from pogema import GridConfig, pogema_v0
from pprint import pprint

class LocalQLearningAgent:
    def __init__(self, eps=0.1, alpha=0.1, gamma=0.9):
        self.q_table = dict()
        self.eps = eps
        self.alpha = alpha
        self.gamma = gamma
        self.prev_state = None
        self.prev_action = None
        self.prev_pos = (3, 3)  # 相对坐标固定为中心
        self.prev_goal_dist = None
        self.strike_time = 1

    def get_state(self, obs_tensor):
        agent_pos = (3, 3)
        goal_pos = np.argwhere(obs_tensor[2] == 1)
        if len(goal_pos) == 0:
            return ("no_goal",)
        gy, gx = goal_pos[0]
        dy, dx = gy - agent_pos[0], gx - agent_pos[1]
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        obstacles = tuple(
            (obs_tensor[0][agent_pos[0] + dy_][agent_pos[1] + dx_]
             if 0 <= agent_pos[0] + dy_ < 7 and 0 <= agent_pos[1] + dx_ < 7 else 1)
            for dy_, dx_ in directions
        )
        return (dy, dx, obstacles)

    def compute_reward(self, obs_tensor, action):
        agent_pos = (3, 3)
        goal_pos = np.argwhere(obs_tensor[2] == 1)

        gy, gx = goal_pos[0]
        new_pos = {
            0: agent_pos,
            1: (agent_pos[0] - 1, agent_pos[1]),
            2: (agent_pos[0] + 1, agent_pos[1]),
            3: (agent_pos[0], agent_pos[1] - 1),
            4: (agent_pos[0], agent_pos[1] + 1),
        }[action]
        new_dist = abs(gy - new_pos[0]) + abs(gx - new_pos[1])

        # 第一次记录目标距离
        if self.prev_goal_dist is None:
            self.prev_goal_dist = abs(gy - agent_pos[0]) + abs(gx - agent_pos[1])
            self.prev_pos = agent_pos
            return 0.0

        # 奖励逻辑
        if new_pos == self.prev_pos:
            reward = -0.2
        elif new_dist < self.prev_goal_dist:
            reward = 0.3
        elif new_dist >= self.prev_goal_dist:
            reward = -0.5
        else:
            reward = -0.1 * self.strike_time  # 平移但距离不变

        self.prev_goal_dist = new_dist
        self.prev_pos = new_pos
        return reward

    def choose_action(self, state, valid_actions):
        if state not in self.q_table:
            self.q_table[state] = [0.0] * 5
        if np.random.rand() < self.eps:
            return np.random.choice(valid_actions)
        q_values = self.q_table[state]
        return max(valid_actions, key=lambda a: q_values[a])

    def update(self, reward, new_state, done, next_action):
        if self.prev_state is None:
            return
        if new_state not in self.q_table:
            self.q_table[new_state] = [0.0] * 5
        q_predict = self.q_table[self.prev_state][next_action]
        q_target = reward if done else reward + self.gamma * max(self.q_table[new_state])
        self.q_table[new_state][next_action] += self.alpha * (q_target - q_predict)

    def act(self, id, obs_tensor, valid_actions, done=False):
        state = self.get_state(obs_tensor)
        state = state[0:2]
        action = self.choose_action(state, valid_actions)
        reward = self.compute_reward(obs_tensor, action)
        self.update(reward, state, done, action)
        self.prev_state = state
        self.prev_action = action
        if id == "player_3":
            print(state)
            pprint(self.q_table, indent=2, sort_dicts=False)
            print(action)
            print(reward)
        return action


class HybridAgent:
    def __init__(self):
        self.rl = LocalQLearningAgent()

    def hybrid_move(self, id, obs_tensor, done=False, ):
        obstacle_map = obs_tensor[0]
        agent_map = obs_tensor[1]
        goal_map = obs_tensor[2]
        agent_pos = (3, 3)

        goal_pos = np.argwhere(goal_map == 1)
        if len(goal_pos) == 0:
            return 0  # 无目标

        # 四个基本方向（dy, dx, action）
        directions = [(-1, 0, 1), (1, 0, 2), (0, -1, 3), (0, 1, 4)]

        valid_actions = []
        for dy_, dx_, a in directions:
            ny, nx = agent_pos[0] + dy_, agent_pos[1] + dx_
            if 0 <= ny < 7 and 0 <= nx < 7 and obstacle_map[ny][nx] == 0 and agent_map[ny][nx] == 0:
                valid_actions.append(a)
        if not valid_actions:
            return 0  # 被障碍物或其他 agent 卡住

        return self.rl.act(id, obs_tensor, valid_actions, done)


# ===== 主运行环境 =====

if __name__ == '__main__':
    seed = 41
    env = pogema_v0(GridConfig(integration="PettingZoo", num_agents=4, obs_radius=3, seed=seed))
    observations = env.reset(seed=seed)
    env.render()
    # 初始化 agent 状态
    agent_map = {agent: HybridAgent() for agent in env.agents}
    last_rewards = {agent: 0 for agent in env.agents}
    done = {agent: False for agent in env.agents}

    while not all(done.values()):
        actions = {}
        for agent in env.agents:
            if done[agent]:
                continue
            obs_tensor = observations[agent]
            actions[agent] = agent_map[agent].hybrid_move(agent, obs_tensor,  done[agent])

        observations, rewards, terminations, truncations, infos = env.step(actions)

        # 保存用于下一步更新的 reward
        last_rewards = rewards
        done = {agent: terminations[agent] or truncations[agent] for agent in env.agents}
        env.render()
