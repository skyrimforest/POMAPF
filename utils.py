import matplotlib.pyplot as plt
import numpy as np

# 示例：3 个算法的 4 种指标
algorithms = ['Baseline', 'A*-FOV', 'RL-Policy', "PPO"]
completion_rate = [0.85, 0.92, 0.88]  # 完成率 (0~1)
makespan = [120, 100, 110]  # Makespan (越小越好)
soc = [1500, 1200, 1300]  # Sum of Costs
collisions = [12, 3, 7]  # 冲突次数 (越小越好)

# 组合数据
metrics = ['完成率', 'Makespan', 'Sum of Costs', '冲突次数']
data = [completion_rate, makespan, soc, collisions]

# 设置柱状图参数
x = np.arange(len(algorithms))  # X轴分组数量
width = 0.2  # 每个柱的宽度

# 创建图形和子图
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制每个指标的柱状图
for i, metric_data in enumerate(data):
    ax.bar(x + i * width, metric_data, width, label=metrics[i])

# 设置 X 轴和标签
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(algorithms, fontsize=12)
ax.set_ylabel('指标数值', fontsize=12)
ax.set_title('多种算法下的 PO-MAPF 性能指标对比', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, linestyle='--', alpha=0.5)

# 展示图表
plt.tight_layout()
plt.show()
