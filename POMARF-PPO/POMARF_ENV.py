'''
@Project ：JiangPro 
@File    ：POMARF_ENV.py
@IDE     ：PyCharm 
@Author  ：Skyrim
@Date    ：2025/6/18 10:33 
'''
from pogema import GridConfig, pogema_v0
import random

# 初始化环境
env = pogema_v0(GridConfig(integration="PettingZoo", num_agents=4, obs_radius=3, seed=random.randint))
observations = env.reset(seed=random.randint)
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

