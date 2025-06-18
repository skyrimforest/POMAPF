'''
@Project ：JiangPro 
@File    ：Greedy.py
@IDE     ：PyCharm 
@Author  ：Skyrim
@Date    ：2025/5/9 17:13 
'''
# 初始化环境
import numpy as np
from pogema import GridConfig, pogema_v0
from collections import deque


class GreedyAgent:
    def __init__(self, memory_size=5):
        self.memory_size = memory_size
        self.position_history = deque(maxlen=memory_size)  # 存储上次的距离
        self.stuck_counter = 0  # 卡死计数器

    def greedy_move(self, id,obs_tensor):
        """改进版贪心策略"""
        obstacle_map = obs_tensor[0]
        agent_map = obs_tensor[1]
        goal_map = obs_tensor[2]
        agent_pos = (3, 3)  # 局部观测中心坐标

        # 检测目标
        goal_pos = np.argwhere(goal_map == 1)
        if len(goal_pos) == 0:
            return 0  # 无目标保持不动

        gy, gx = goal_pos[0]  # 目标相对坐标
        dy, dx = gy - agent_pos[0], gx - agent_pos[1]

        # 动作优先级：先尝试靠近目标的方向
        base_directions = [(0, -1, 3), (1,0, 2), (-1, 0, 1), (0,1, 4)]  # (dy, dx, action)
        dirction_map={
            4:(0,1),  # right
            3:(0,-1), # left
            2:(1,0),  # down
            1:(-1,0), # up
        }
        # 计算各方向与目标的曼哈顿距离差值
        def direction_score(d):
            return (dy - d[0]) ** 2 + (dx - d[1]) ** 2

        # 按优先级排序（距离差越小优先级越高）
        directions = sorted(base_directions, key=direction_score)

        # 尝试所有可能方向
        valid_actions = []
        for ddy, ddx, action in directions:
            ny, nx = agent_pos[0] + ddy, agent_pos[1] + ddx
            if 0 <= ny < 7 and 0 <= nx < 7 and obstacle_map[ny][nx] == 0 and agent_map[ny][nx] == 0:
                valid_actions.append(action)

        # 无可用动作时保持不动
        if not valid_actions:
            return 0


        # 防振荡机制：如果最近几步在重复位置，强制改变方向
        if len(self.position_history) == self.memory_size:
            if len(set(self.position_history)) < 3:  # 位置重复度高
                self.stuck_counter += 1
                if self.stuck_counter > 2:  # 连续卡死时
                    # 尝试与当前最佳方向垂直的动作
                    alt_directions = [a for a in valid_actions if a != directions[0][2]]
                    if alt_directions:
                        target_action=np.random.choice(alt_directions)
                        self.position_history.append(direction_score(dirction_map[target_action]))
                        return target_action
            else:
                self.stuck_counter = 0

        # 记录当前位置
        self.position_history.append(direction_score(dirction_map[valid_actions[0]]))

        # if id=="player_3":
        #     print(agent_map)
        #     print(obstacle_map)
        #     print(valid_actions)
        #     print(directions)
        #     print(self.position_history)
        #     print(self.stuck_counter)

        # 默认返回最优可行动作
        return valid_actions[0]


# 初始化环境
seed = 43
env = pogema_v0(GridConfig(integration="PettingZoo", num_agents=4, obs_radius=3, seed=23))
observations = env.reset()
print(env.agents)

env.render()
agent_map = {agent: GreedyAgent() for agent in env.agents}
done = {agent: False for agent in env.agents}
cnt = 0
while not all(done.values()):
    actions = {}
    for agent in env.agents:
        if done[agent]:
            continue
        cnt += 1
        obs_tensor = observations[agent]  # shape: (3, 7, 7)
        actions[agent] = agent_map[agent].greedy_move(agent,obs_tensor)
    observations, rewards, terminations, truncations, infos = env.step(actions)
    done = {agent: terminations[agent] or truncations[agent] for agent in env.agents}
    env.render()
