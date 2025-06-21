'''
@Project ：JiangPro 
@File    ：show_graph.py
@IDE     ：PyCharm 
@Author  ：Skyrim
@Date    ：2025/6/19 11:47 
'''
import hiddenlayer as hl
import torch
from PPO import PPO

action_std = 0.6  # starting std for action distribution (Multivariate Normal)
action_std_decay_rate = 0.05  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
min_action_std = 0.1  # minimum action_std (stop decay after action_std <= min_action_std)
action_std_decay_freq = int(2.5e5)
K_epochs = 80  # update policy for K epochs in one PPO update

eps_clip = 0.2  # clip parameter for PPO
gamma = 0.99  # discount factor

lr_actor = 0.0003  # learning rate for actor network
lr_critic = 0.001  # learning rate for critic network
# 创建可视化图
state_dim=49
action_dim=4
has_continuous_action_space=False
model = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,
                action_std)
example_input=torch.randn(1, state_dim)
print(model.policy)
graph = hl.build_graph(model.policy, example_input)
graph.save("model_hiddenlayer", format="png")
