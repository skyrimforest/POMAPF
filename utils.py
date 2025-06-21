import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.use('TkAgg')  # Use 'TkAgg' or 'Qt5Agg' backend if available
sns.set(style="whitegrid", palette="muted", font_scale=1.2)

# Algorithm names
algorithms = ['Greedy', 'A*-FOV', 'RL-Policy', 'PPO']

# Simulated data (replace with actual experiment results)
completion_rate = [0.62, 0.85, 0.92, 0.95]
makespan = [85, 63, 48, 45]
sum_of_costs = [210, 180, 160, 150]
collision_count = [22, 8, 3, 2]

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Performance Comparison of Algorithms in POMARF", fontsize=16)

# Define metrics (title, data, unit, axis)
metrics = [
    ("Completion Rate", completion_rate, "%", axs[0, 0]),
    ("Makespan", makespan, "Steps", axs[0, 1]),
    ("Sum of Costs", sum_of_costs, "Total Steps", axs[1, 0]),
    ("Number of Collisions", collision_count, "Times", axs[1, 1]),
]

# Plotting each metric
for title, data, unit, ax in metrics:
    sns.barplot(x=algorithms, y=data, ax=ax, hue=algorithms, dodge=False)
    ax.set_title(title)
    ax.set_ylabel(unit)
    ax.set_xlabel("")
    ax.set_ylim(0, max(data) * 1.2)
    ax.bar_label(ax.containers[0], fmt="%.2f" if "Rate" in title else "%d")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
