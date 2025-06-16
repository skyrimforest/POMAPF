'''
@Project ：JiangPro 
@File    ：Astar.py
@IDE     ：PyCharm 
@Author  ：Skyrim
@Date    ：2025/6/14 21:24 
'''
import heapq
import numpy as np
from pogema import GridConfig, pogema_v0
from collections import deque

class AStarAgent:
    def __init__(self):
        self.agent_pos = (3, 3)  # 7x7局部视野中心

    def heuristic(self, a, b):
        # 曼哈顿距离
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def a_star(self, start, goal, obstacle_map):
        h, w = obstacle_map.shape
        open_set = []
        heapq.heappush(open_set, (0 + self.heuristic(start, goal), 0, start))
        came_from = {}
        g_score = {start: 0}

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while open_set:
            _, cost, current = heapq.heappop(open_set)

            if current == goal:
                # 回溯路径
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path

            for dy, dx in directions:
                neighbor = (current[0] + dy, current[1] + dx)
                if 0 <= neighbor[0] < h and 0 <= neighbor[1] < w:
                    if obstacle_map[neighbor] != 0:
                        continue  # 障碍物
                    tentative_g = g_score[current] + 1
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f = tentative_g + self.heuristic(neighbor, goal)
                        heapq.heappush(open_set, (f, tentative_g, neighbor))
        return []  # 无法到达

    def get_action(self, obs_tensor):
        obstacle_map = obs_tensor[0]
        agent_map = obs_tensor[1]
        goal_map = obs_tensor[2]

        start = self.agent_pos
        goal_pos = np.argwhere(goal_map == 1)

        if len(goal_pos) == 0:
            return 0  # 无目标，不动

        goal = tuple(goal_pos[0])

        path = self.a_star(start, goal, obstacle_map)
        if not path:
            return 0  # 无路径，不动

        next_pos = path[0]  # 下一步

        dy = next_pos[0] - start[0]
        dx = next_pos[1] - start[1]
        move_map = {
            (-1, 0): 1,  # 上
            (1, 0): 2,   # 下
            (0, -1): 3,  # 左
            (0, 1): 4    # 右
        }
        return move_map.get((dy, dx), 0)


# 初始化环境
seed = 43
env = pogema_v0(GridConfig(integration="PettingZoo", num_agents=4, obs_radius=3, seed=seed))
observations = env.reset(seed=seed)
env.render()

agent_map = {agent: AStarAgent() for agent in env.agents}
done = {agent: False for agent in env.agents}

while not all(done.values()):
    actions = {}
    for agent in env.agents:
        if done[agent]:
            continue
        obs_tensor = observations[agent]
        actions[agent] = agent_map[agent].get_action(obs_tensor)
    observations, rewards, terminations, truncations, infos = env.step(actions)
    done = {agent: terminations[agent] or truncations[agent] for agent in env.agents}
    env.render()
